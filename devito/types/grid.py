from abc import ABC
from collections import namedtuple
from functools import cached_property

import numpy as np
from sympy import prod

from devito import configuration
from devito.data import LEFT, RIGHT
from devito.logger import warning
from devito.mpi import Distributor, MPI, SubDistributor
from devito.tools import ReducerMap, as_tuple
from devito.types.args import ArgProvider
from devito.types.basic import Scalar
from devito.types.dense import Function
from devito.types.utils import DimensionTuple
from devito.types.dimension import (Dimension, SpaceDimension, TimeDimension,
                                    Spacing, SteppingDimension, SubDimension,
                                    MultiSubDimension, DefaultDimension)
from devito.deprecations import deprecations

__all__ = ['Grid', 'SubDomain', 'SubDomainSet']


GlobalLocal = namedtuple('GlobalLocal', 'glb loc')


class CartesianDiscretization(ABC):

    """
    Abstract base class for objects representing discretizations of n-dimensional
    physical domains by congruent parallelotopes (e.g., "tiles" or "bricks").
    """

    is_Grid = False
    is_SubDomain = False
    is_MultiSubDomain = False

    def __init__(self, shape=None, dimensions=None, dtype=None):
        self._shape = as_tuple(shape)
        self._dimensions = as_tuple(dimensions)
        self._dtype = dtype

    @property
    def shape(self):
        """Shape of the physical domain."""
        return DimensionTuple(*self._shape, getters=self.dimensions)

    @property
    def dimensions(self):
        """Spatial dimensions of the computational domain."""
        return self._dimensions

    @property
    def dim(self):
        """Problem dimension, or number of spatial dimensions."""
        return len(self.dimensions)

    @property
    def dtype(self):
        """
        Data type inherited by all Functions defined on this CartesianDiscretization.
        """
        return self._dtype

    @property
    def root(self):
        return self


class Grid(CartesianDiscretization, ArgProvider):

    """
    A cartesian grid that encapsulates a computational domain over which
    to discretize a Function.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the computational domain in grid points.
    extent : tuple of values interpretable as dtype, default=unit box of extent 1m
    in all dimensions.
        Physical extent of the domain in m.
    origin : tuple of values interpretable as dtype, default=0.0 in all dimensions
        Physical coordinate of the origin of the domain.
    dimensions : tuple of SpaceDimension, optional
        The dimensions of the computational domain encapsulated by this Grid.
    time_dimension : TimeDimension, optional
        The dimension used to define time in a `TimeFunction` created from
        this Grid.
    dtype : data-type, default=np.float32
        Any object that can be interpreted as a numpy data type, used as default
        data type to be inherited by all Functions created from this Grid.
    subdomains : tuple of SubDomain, optional
        If no subdomains are specified, the Grid only defines the two default
        subdomains ``interior`` and ``domain``.
    comm : MPI communicator, optional
        The set of processes over which the grid is distributed. Only relevant in
        case of MPI execution.

    Examples
    --------
    >>> from devito import Grid, Function
    >>> grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
    >>> f = Function(name='f', grid=grid)
    >>> f.shape
    (4, 4)
    >>> f.dimensions
    (x, y)
    >>> f.dtype
    <class 'numpy.float32'>

    In a Function, the domain defined by a Grid is often surrounded by a "halo
    region", which guarantees the correctness of stencil updates nearby the
    domain boundary. However, the size of the halo region does *not* depend on
    the Grid; for more information, refer to ``Function.__doc__``.

    >>> f.shape_with_halo
    (6, 6)

    Notes
    -----
    A Grid encapsulates the topology and geometry information of the
    computational domain that a Function can be discretized on.  As such, it
    defines and provides the physical coordinate information of the logical
    cartesian grid underlying the discretized Functions.  For example, the
    conventions for defining the coordinate space in 2D are:

        .. code-block:: python

                       x
             |----------------------->
             |  origin
             |     o------------o
             |     |            |
             |     |            |
             |     |   DOMAIN   | extent[1]
         y   |     |            |
             |     |            |
             |     |  extent[0] |
             |     o------------o
             |             origin + extent
             |
             v
    """

    is_Grid = True

    _default_dimensions = ('x', 'y', 'z')

    def __init__(self, shape, extent=None, origin=None, dimensions=None,
                 time_dimension=None, dtype=np.float32, subdomains=None,
                 comm=None, topology=None):
        shape = as_tuple(shape)

        # Create or pull the SpaceDimensions
        if dimensions is None:
            ndim = len(shape)
            assert ndim <= 3
            dim_names = self._default_dimensions[:ndim]
            dim_spacing = tuple(Spacing(name='h_%s' % n, dtype=dtype, is_const=True)
                                for n in dim_names)
            dimensions = tuple(SpaceDimension(name=n, spacing=s)
                               for n, s in zip(dim_names, dim_spacing))
        else:
            for d in dimensions:
                if not d.is_Space:
                    raise ValueError("Cannot create Grid with Dimension `%s` "
                                     "since it's not a SpaceDimension" % d)
                if d.is_Derived and not d.is_Conditional:
                    raise ValueError("Cannot create Grid with derived Dimension `%s` "
                                     "of type `%s`" % (d, type(d)))
            dimensions = dimensions

        super().__init__(shape, dimensions, dtype)

        # Create a Distributor, used internally to implement domain decomposition
        # by all Functions defined on this Grid
        topology = topology or configuration['topology']
        if topology:
            if len(topology) == len(self.shape):
                self._topology = topology
            else:
                warning("Ignoring the provided topology `%s` as it "
                        "is incompatible with the grid shape `%s`" %
                        (topology, self.shape))
                self._topology = None
        else:
            self._topology = None
        self._distributor = Distributor(shape, dimensions, comm, self._topology)

        # The physical extent
        extent = as_tuple(extent or tuple(1. for _ in self.shape))
        self._extent = tuple(dtype(e) for e in extent)

        # The origin of the grid
        origin = as_tuple(origin or tuple(0. for _ in self.shape))
        self._origin = tuple(dtype(o) for o in origin)
        self._origin_symbols = tuple(Scalar(name='o_%s' % d.name, dtype=dtype,
                                            is_const=True)
                                     for d in self.dimensions)

        # Sanity check
        assert (self.dim == len(self.origin) == len(self.extent) == len(self.spacing))

        # Store or create default symbols for time and stepping dimensions
        if time_dimension is None:
            spacing = Scalar(name='dt', dtype=dtype, is_const=True)
            self._time_dim = TimeDimension(name='time', spacing=spacing)
            self._stepping_dim = SteppingDimension(name='t', parent=self.time_dim)
        elif isinstance(time_dimension, TimeDimension):
            self._time_dim = time_dimension
            self._stepping_dim = SteppingDimension(name='%s_s' % self.time_dim.name,
                                                   parent=self.time_dim)
        else:
            raise ValueError("`time_dimension` must be None or of type TimeDimension")

        # Initialize SubDomains for legacy interface
        if subdomains is not None:
            deprecations.subdomain_warn
        self._subdomains = tuple(i for i in (Domain(), Interior(), *as_tuple(subdomains)))
        for i in self._subdomains:
            i.__subdomain_finalize_legacy__(self)

    def __repr__(self):
        return "Grid[extent=%s, shape=%s, dimensions=%s]" % (
            self.extent, self.shape, self.dimensions
        )

    @property
    def extent(self):
        """Physical extent of the domain in m."""
        return self._extent

    @property
    def origin(self):
        """Physical coordinates of the domain origin."""
        return self._origin

    @property
    def origin_symbols(self):
        """Symbols representing the grid origin in each SpaceDimension."""
        return self._origin_symbols

    @property
    def origin_map(self):
        """Map between origin symbols and their values."""
        return dict(zip(self.origin_symbols, self.origin))

    @property
    def origin_ioffset(self):
        """Offset index of the local (per-process) origin from the domain origin."""
        grid_origin = [min(i) for i in self.distributor.glb_numb]
        assert len(grid_origin) == len(self.spacing)
        return DimensionTuple(*grid_origin, getters=self.dimensions)

    @property
    def origin_offset(self):
        """Physical offset of the local (per-process) origin from the domain origin."""
        return DimensionTuple(*[i*h for i, h in zip(self.origin_ioffset, self.spacing)],
                              getters=self.dimensions)

    @property
    def time_dim(self):
        """Time dimension associated with this Grid."""
        return self._time_dim

    @property
    def stepping_dim(self):
        """Stepping dimension associated with this Grid."""
        return self._stepping_dim

    @property
    def subdomains(self):
        """The SubDomains defined in this Grid."""
        return {i.name: i for i in self._subdomains}

    @property
    def interior(self):
        """The interior SubDomain of the Grid."""
        return self.subdomains['interior']

    @property
    def volume_cell(self):
        """Volume of a single cell e.g  h_x*h_y*h_z in 3D."""
        return prod(d.spacing for d in self.dimensions).subs(self.spacing_map)

    @property
    def spacing(self):
        """Spacing between grid points in m."""
        spacing = (np.array(self.extent) / (np.array(self.shape) - 1)).astype(self.dtype)
        return as_tuple(spacing)

    @cached_property
    def spacing_symbols(self):
        """Symbols representing the grid spacing in each SpaceDimension."""
        return as_tuple(d.root.spacing for d in self.dimensions)

    @cached_property
    def spacing_map(self):
        """Map between spacing symbols and their values for each SpaceDimension."""
        mapper = {}
        for d, s in zip(self.dimensions, self.spacing):
            if d.is_Conditional:
                # Special case subsampling: `Grid.dimensions` -> (xb, yb, zb)`
                # where `xb, yb, zb` are ConditionalDimensions whose parents
                # are SpaceDimensions
                mapper[d.root.spacing] = s/self.dtype(d.factor)
            elif d.is_Space:
                # Typical case: `Grid.dimensions` -> (x, y, z)` where `x, y, z` are
                # the SpaceDimensions
                mapper[d.spacing] = s
            else:
                assert False

        return mapper

    @property
    def shape_local(self):
        """Shape of the local (per-process) physical domain."""
        return self._distributor.shape

    @property
    def size_map(self):
        """Map between SpaceDimensions and their global/local size."""
        return {d: GlobalLocal(g, l)
                for d, g, l in zip(self.dimensions, self.shape, self.shape_local)}

    @property
    def topology(self):
        """The topology used for decomposing the CartesianDiscretization."""
        return self._topology

    @property
    def distributor(self):
        """The Distributor used for decomposing the CartesianDiscretization."""
        return self._distributor

    @property
    def comm(self):
        """The MPI communicator inherited from the distributor."""
        return self._distributor.comm

    def is_distributed(self, dim):
        """
        True if `dim` is a distributed Dimension for this CartesianDiscretization,
        False otherwise.
        """
        return any(dim is d for d in self.distributor.dimensions)

    @cached_property
    def _arg_names(self):
        ret = []
        ret.append(self.time_dim.spacing.name)
        ret.extend([i.name for i in self.origin_map])
        for i in self.spacing_map:
            try:
                ret.append(i.name)
            except AttributeError:
                # E.g., {n*h_x: v} (the case of ConditionalDimension)
                ret.extend([a.name for a in i.free_symbols])
        return tuple(ret)

    def _arg_defaults(self):
        """A map of default argument values defined by this Grid."""
        args = ReducerMap()

        # Dimensions size
        for k, v in self.size_map.items():
            args.update(k._arg_defaults(_min=0, size=v.loc))

        # Dimensions spacing
        args.update({k.name: v for k, v in self.spacing_map.items()})

        # Grid origin
        args.update({k.name: v for k, v in self.origin_map.items()})

        # MPI-related objects
        if self.distributor.is_parallel:
            distributor = self.distributor
            args[distributor._obj_comm.name] = distributor._obj_comm.value
            args[distributor._obj_neighborhood.name] = distributor._obj_neighborhood.value

        return args

    def _arg_values(self, **kwargs):
        values = dict(self._arg_defaults())

        # Override spacing and origin if necessary
        values.update({i: kwargs[i] for i in self._arg_names if i in kwargs})

        return values

    def __getstate__(self):
        state = self.__dict__.copy()
        # A Distributor wraps an MPI communicator, which can't and shouldn't be pickled
        state.pop('_distributor')
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._distributor = Distributor(self.shape, self.dimensions, MPI.COMM_SELF)


class AbstractSubDomain(CartesianDiscretization):

    """
    Abstract base class for subdomains.
    """

    is_SubDomain = True

    name = None
    """A unique name for the SubDomain."""

    def __init__(self, *args, **kwargs):
        if self.name is None:
            self.name = self.__class__.__name__

        # All other attributes get initialized upon `__subdomain_finalize__`
        super().__init__()

        self._distributor = None

        self._grid = kwargs.get('grid')
        if self.grid:
            self.__subdomain_finalize__()

    def __subdomain_finalize__(self):
        """
        Finalize the subdomain initialization.

        Notes
        -----
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def __subdomain_finalize_legacy__(self, grid):
        """
        Finalize the subdomain initialization.

        (Backward-compatible version for legacy SubDomain API)

        Notes
        -----
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, AbstractSubDomain):
            return False
        return (self.name == other.name and
                self.dimensions == other.dimensions and
                self.shape == other.shape and
                self.dtype == other.dtype)

    def __hash__(self):
        return hash((self.name, self.dimensions, self.shape, self.dtype))

    def __str__(self):
        return "%s[%s%s]" % (self.__class__.__name__, self.name, self.dimensions)

    __repr__ = __str__

    def union(self, other):
        """
        Return the union of two subdomains as a new subdomain.
        """
        raise NotImplementedError

    def intersection(self, other):
        """
        Return the intersection of two subdomains as a new subdomain.
        """
        raise NotImplementedError

    def difference(self, other):
        """
        Return the difference of two subdomains as a new subdomain.
        """
        raise NotImplementedError

    @property
    def dimension_map(self):
        return {d.root: d for d in self.dimensions}

    @property
    def grid(self):
        return self._grid

    @property
    def root(self):
        return self.grid.root

    @property
    def time_dim(self):
        if self.grid:
            return self.grid.time_dim
        raise AttributeError("SubDomain has no Grid, and thus no time dimension")

    @property
    def stepping_dim(self):
        if self.grid:
            return self.grid.stepping_dim
        raise AttributeError("SubDomain has no Grid, and thus no stepping dimension")

    @property
    def distributor(self):
        """The Distributor used for MPI-decomposing the CartesianDiscretization."""
        return self._distributor

    def is_distributed(self, dim):
        """
        True if `dim` is a distributed Dimension for this CartesianDiscretization,
        False otherwise.
        """
        if self.grid:
            return any(dim is d for d in self.distributor.dimensions)
        return False

    @property
    def comm(self):
        """The MPI communicator inherited from the distributor."""
        if self.grid:
            return self.grid.comm
        raise ValueError("`SubDomain` %s has no `Grid` attached and thus no `comm`"
                         % self.name)

    def _arg_values(self, **kwargs):
        try:
            return self.grid._arg_values(**kwargs)
        except AttributeError:
            raise AttributeError("%s is not attached to a Grid and has no _arg_values"
                                 % self)


class SubDomain(AbstractSubDomain):

    """
    Base class to define Grid subdomains.

    To create a new SubDomain, all one needs to do is overriding :meth:`define`.
    This method takes as input a set of Dimensions and produce a mapper

        ``M : Dimensions -> {d, ('left', N), ('middle', N, M), ('right', N)}``

    so that:

        * If ``M(d) = d``, then the SubDomain spans the entire Dimension ``d``.
        * If ``M(d) = ('left', N)``, then the SubDomain spans a contiguous
          region of ``N`` points starting at ``d``\'s left extreme.
        * ``M(d) = ('right', N)`` is analogous to the case above.
        * If ``M(d) = ('middle', N, M)``, then the SubDomain spans a contiguous
          region of ``d_size - (N + M)`` points starting at ``N`` and finishing
          at ``d_sizeM - M``.

    Examples
    --------
    An "Inner" SubDomain, which spans the entire domain except for an exterior
    boundary region of ``thickness=3``, can be implemented as follows

    >>> from devito import SubDomain
    >>> class Inner(SubDomain):
    ...     name = 'inner'
    ...     def define(self, dimensions):
    ...         return {d: ('middle', 3, 3) for d in dimensions}

    Like before, but now spanning the entire ``y`` Dimension of a three-dimensional
    grid

    >>> class InnerY(SubDomain):
    ...     name = 'inner_y'
    ...     def define(self, dimensions):
    ...         x, y, z = dimensions
    ...         return {x: ('middle', 3, 3), y: y, z: ('middle', 3, 3)}

    See Also
    --------
    Domain : An example of preset SubDomain.
    Interior : An example of preset Subdomain.

    Notes
    -----
    SubDomains are the only way to harness the benefits of domain decomposition,
    especially when defining BCs.
    """

    def __subdomain_finalize__(self):
        self.__subdomain_finalize_legacy__(self.grid)
        self._distributor = SubDistributor(self)

        # Intervals of form Interval(n, n) automatically become FiniteSet
        # Add one to end as intervals are in terms of indices (inclusive of endpoints)
        # Empty interval corresponds to a size of zero
        self._shape_local = tuple(0 if i.is_empty else i.end-i.start + 1 if i.is_Interval
                                  else 1 for i in self.distributor.intervals)

    def __subdomain_finalize_legacy__(self, grid):
        # Create the SubDomain's SubDimensions
        sub_dimensions = []
        sdshape = []
        for k, v, s in zip(self.define(grid.dimensions).keys(),
                           self.define(grid.dimensions).values(), grid.shape):
            if isinstance(v, Dimension):
                sub_dimensions.append(v)
                sdshape.append(s)
            else:
                try:
                    # Case ('middle', int, int)
                    side, ltkn, rtkn = v
                    if side != 'middle':
                        raise ValueError(f"Expected side 'middle', not `{side}`")
                    sub_dimensions.append(SubDimension.middle(f'i{k.name}',
                                                              k, ltkn, rtkn))
                    thickness = s-ltkn-rtkn
                    sdshape.append(thickness)
                except ValueError:
                    side, thickness = v
                    constructor = {'left': SubDimension.left,
                                   'right': SubDimension.right}.get(side, None)
                    if constructor is None:
                        raise ValueError(f"Expected sides 'left|right', not `{side}`")

                    if s - thickness < 0:
                        raise ValueError(f"Maximum thickness of dimension {k.name} "
                                         f"is {s}, not {thickness}")
                    sub_dimensions.append(constructor(f'i{k.name}', k, thickness))
                    sdshape.append(thickness)

        self._shape = tuple(sdshape)
        self._dimensions = tuple(sub_dimensions)
        self._dtype = grid.dtype

    @property
    def shape_local(self):
        return self._shape_local

    @property
    def size_map(self):
        """Map between SpaceDimensions and their global/local size."""
        return {d: GlobalLocal(g, l)
                for d, g, l in zip(self.dimensions, self.shape, self.shape_local)}

    def define(self, dimensions):
        """
        Parametrically describe the SubDomain w.r.t. a generic Grid.

        Notes
        -----
        This method should be overridden by each SubDomain subclass. For more
        information, refer to ``SubDomain.__doc__``.
        """
        raise NotImplementedError

    @cached_property
    def _arg_names(self):
        try:
            ret = self.grid._arg_names
        except AttributeError:
            msg = f"{self} is not attached to a Grid and has no _arg_names"
            raise AttributeError(msg)

        # Names for SubDomain thicknesses
        thickness_names = tuple([k.name for k in d._thickness_map]
                                for d in self.dimensions if d.is_Sub)

        ret += tuple(thickness_names)

        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        # A SubDistributor wraps an MPI communicator, which can't and shouldn't be pickled
        state.pop('_distributor', None)
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        if self.grid:
            self._distributor = SubDistributor(self)


class MultiSubDomain(AbstractSubDomain):

    """
    Abstract base class for types representing groups of SubDomains.
    """

    is_MultiSubDomain = True

    def __hash__(self):
        # There is no possibility for two MultiSubDomains to ever hash the same since
        # they are by construction unique and different from each other
        return id(self)

    @classmethod
    def _bounds_glb_to_loc(cls, dec, m, M):
        """
        Translate a SubDomain global bounds, that is thicknesses, into local bounds.
        """

        # There are infinite ways to set `bounds_m` and `bounds_M` to set the size
        # of an MPI-decomposed local subdomain to 0 iterations, meaning that after
        # domain decomposition and for the given thicknesses there are 0 iterations
        # left to execute. However, due to issue #1766, we only have one choice, or
        # MultiSubDomains might break on GPUs
        NOITER = (dec.loc_abs_max, 1)

        bounds_m = np.zeros(m.shape, dtype=m.dtype)
        bounds_M = np.zeros(m.shape, dtype=m.dtype)
        for j in range(m.size):
            lmin = dec.glb_min + m[j]
            lmax = dec.glb_max - M[j]

            # Check if the subdomain doesn't intersect with the decomposition
            if lmin < dec.loc_abs_min and lmax < dec.loc_abs_min:
                bounds_m[j], bounds_M[j] = NOITER
                continue
            if lmin > dec.loc_abs_max and lmax > dec.loc_abs_max:
                bounds_m[j], bounds_M[j] = NOITER
                continue

            if lmin < dec.loc_abs_min:
                bounds_m[j] = 0
            elif lmin > dec.loc_abs_max:
                bounds_m[j], bounds_M[j] = NOITER
                continue
            else:
                bounds_m[j] = dec.index_glb_to_loc(m[j], LEFT)

            if lmax < dec.loc_abs_min:
                bounds_m[j], bounds_M[j] = NOITER
                continue
            elif lmax >= dec.loc_abs_max:
                bounds_M[j] = 0
            else:
                bounds_M[j] = dec.index_glb_to_loc(M[j], RIGHT)

        return bounds_m, bounds_M


class SubDomainSet(MultiSubDomain):
    """
    Class to define a set of N (a positive integer) subdomains.

    Parameters
    ----------
    **kwargs
        * N : int
            Number of subdomains.
        * bounds : tuple
            Tuple of numpy int32 arrays representing the bounds of
            each subdomain.

    Examples
    --------
    Set up an iterate upon a set of two subdomains:

    >>> import numpy as np
    >>> from devito import Grid, Function, Eq, Operator, SubDomainSet
    >>> Nx = 10
    >>> Ny = Nx
    >>> n_domains = 2

    Create a 'SubDomainSet object':

    >>> class MySubdomains(SubDomainSet):
    ...     name = 'mydomains'

    Set the bounds of the subdomains. The required format is:
    (xm, xM, ym, yM, ...) where xm is a vector specifying
    the number of grid points inwards from the 'left' boundary in the
    first grid dimension that each subdomain starts. xM is a vector
    specifying the number of grid points inwards from the 'right' of
    the domain in the first grid dimension that each subdomain ends.
    ym and yM are the equivalents for the second grid dimension.

    >>> xm = np.array([1, Nx/2+1], dtype=np.int32)
    >>> xM = np.array([Nx/2+1, 1], dtype=np.int32)

    Along a dimension where all bounds are the same we can use the
    following shorthand:

    >>> ym = 1 # which is equivalent to 'np.array([1, 1], dtype=np.int32)'
    >>> yM = 1

    Combine the data into the required form:

    >>> bounds = (xm, xM, ym, yM)

    Create our set of subdomains passing the number of domains and the
    bounds:

    >>> my_sd = MySubdomains(N=n_domains, bounds=bounds)

    Create a grid and iterate a function within the defined subdomains:

    >>> grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd, ))
    >>> f = Function(name='f', grid=grid, dtype=np.int32)
    >>> eq = Eq(f, f+1, subdomain=grid.subdomains['mydomains'])
    >>> op = Operator(eq)
    >>> summary = op.apply()
    >>> f.data
    Data([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    """

    def __init__(self, **kwargs):
        self._n_domains = kwargs.get('N', 1)
        self._global_bounds = kwargs.get('bounds', None)
        super().__init__(**kwargs)

        try:
            self.implicit_dimension
            warning("`implicit_dimension` is deprecated. You may safely remove it "
                    "from the class definition")
        except AttributeError:
            pass

    def __subdomain_finalize_core__(self, grid):
        self._dtype = grid.dtype

        # Compute the SubDomainSet shapes
        global_bounds = []
        for i in self._global_bounds:
            if isinstance(i, int):
                global_bounds.append(np.full(self._n_domains, i, dtype=np.int32))
            else:
                global_bounds.append(i)
        d_m = global_bounds[0::2]
        d_M = global_bounds[1::2]
        shapes = []
        for i in range(self._n_domains):
            dshape = []
            for s, m, M in zip(grid.shape, d_m, d_M):
                assert(m.size == M.size)
                dshape.append(s-m[i]-M[i])
            shapes.append(as_tuple(dshape))
        self._shape = as_tuple(shapes)

        if grid.distributor and grid.distributor.is_parallel:
            # Now create local bounds based on distributor
            processed = []
            for dec, m, M in zip(grid.distributor.decomposition, d_m, d_M):
                processed.extend(self._bounds_glb_to_loc(dec, m, M))
            self._local_bounds = as_tuple(processed)
        else:
            # Not distributed and hence local and global bounds are
            # equivalent.
            self._local_bounds = self._global_bounds

        # Sanity check
        if len(self._local_bounds) != 2*len(grid.dimensions):
            raise ValueError("Left and right bounds must be supplied for each dimension")

        # Associate the `_local_bounds` to suitable symbolic objects that the
        # compiler can use to generate code

        # Dimensions with identical names hash the same, hence tag them with the
        # SubDomainSet ID to make them unique so they can be used to key a dictionary
        # of replacements without risking overwriting.
        i_dim = Dimension('n_%s' % str(id(self)))
        d_dim = DefaultDimension(name='d', default_value=2*grid.dim)
        sd_func = Function(name=self.name, grid=self._grid,
                           shape=(self._n_domains, 2*grid.dim),
                           dimensions=(i_dim, d_dim), dtype=np.int32)

        dimensions = []
        for i, d in enumerate(grid.dimensions):
            # Check if shorthand notation has been provided:
            for j in range(2):
                idx = 2*i + j
                sd_func.data[:, idx] = self._local_bounds[idx]

            dimensions.append(MultiSubDimension(
                'i%s' % d.name, d, None, functions=sd_func,
                bounds_indices=(2*i, 2*i+1), implicit_dimension=i_dim
            ))

        self._dimensions = tuple(dimensions)

    def __subdomain_finalize__(self):
        self.__subdomain_finalize_core__(self.grid)

    def __subdomain_finalize_legacy__(self, grid):
        self._grid = grid
        self.__subdomain_finalize_core__(self.grid)

    @property
    def n_domains(self):
        return self._n_domains

    @property
    def bounds(self):
        return self._local_bounds


# Preset SubDomains


class Domain(SubDomain):

    """
    The entire computational domain (== boundary + interior).
    """

    name = 'domain'

    def define(self, dimensions):
        return dict(zip(dimensions, dimensions))


class Interior(SubDomain):

    """
    The interior of the computational domain (i.e., boundaries are excluded).
    """

    name = 'interior'

    def define(self, dimensions):
        return {d: ('middle', 1, 1) for d in dimensions}


preset_subdomains = [Domain, Interior]
npresets = len(preset_subdomains)
