import pystencils as ps
import sympy as sp
from pystencils.fd.derivation import (
    FiniteDifferenceStaggeredStencilDerivation as FDS,
    FiniteDifferenceStencilDerivation as FD,
)
import itertools
from collections import defaultdict
from collections.abc import Iterable


def get_access_and_direction(term):
    direction1 = term.args[1]
    if isinstance(term.args[0], ps.Field.Access):  # first derivative
        access = term.args[0]
        direction = (direction1,)
    elif isinstance(term.args[0], ps.fd.Diff):  # nested derivative
        if isinstance(term.args[0].args[0], ps.fd.Diff):  # third or higher derivative
            raise ValueError("can only handle first and second derivatives")
        elif not isinstance(term.args[0].args[0], ps.Field.Access):
            raise ValueError("can only handle derivatives of field accesses")

        access, direction2 = term.args[0].args[:2]
        direction = (direction1, direction2)
    else:
        raise NotImplementedError(
            f"can only deal with derivatives of field accesses, "
            f"but not {type(term.args[0])}; expansion of derivatives probably failed"
        )
    return access, direction


class FVM1stOrder:
    """Finite-volume discretization

    Args:
        field: the field with the quantity to calculate, e.g. a concentration
        flux: a list of sympy expressions that specify the flux, one for each cartesian direction
        source: a list of sympy expressions that specify the source
    """

    def __init__(self, field: ps.field.Field, flux=0, source=0):
        def normalize(f, shape):
            shape = tuple(s for s in shape if s != 1)
            if not shape:
                shape = None

            if (
                isinstance(f, sp.Array)
                or isinstance(f, Iterable)
                or isinstance(f, sp.Matrix)
            ):
                return sp.Array(f, shape)
            else:
                return sp.Array([f] * (sp.Mul(*shape) if shape else 1))

        self.c = field
        self.dim = self.c.spatial_dimensions
        self.j = normalize(flux, (self.dim,) + self.c.index_shape)
        self.q = normalize(source, self.c.index_shape)

    def discrete_flux(self, flux: ps.StaggeredField):
        """Return a list of assignments for the discrete fluxes

        Args:
            flux_field: a staggered field to which the fluxes should be assigned
        """

        assert isinstance(flux, ps.StaggeredField)

        num = 0

        def discretize(term, neighbor):
            nonlocal num
            if isinstance(term, sp.Matrix):
                nw = term.applyfunc(lambda t: discretize(t, neighbor))
                return nw
            elif isinstance(term, ps.field.Field.Access):
                avg = (term.get_shifted(*neighbor) + term) * sp.Rational(1, 2)
                return avg
            elif isinstance(term, ps.fd.Diff):
                access, direction = get_access_and_direction(term)

                fds = FDS(
                    neighbor,
                    access.field.spatial_dimensions,
                    direction,
                    free_weights_prefix=(
                        f"fvm_free_{num}"
                        if sp.Matrix(neighbor).dot(neighbor) > 2
                        else None
                    ),
                )
                num += 1
                return fds.apply(access)

            if term.args:
                new_args = [discretize(a, neighbor) for a in term.args]
                return term.func(*new_args)
            else:
                return term

        fluxes = self.j.applyfunc(ps.fd.derivative.expand_diff_full)
        # is_vector is True when the flux has per-component entries (multi-species / vector field)
        is_vector = len(self.j.shape) > 1
        if is_vector:
            n_vec = self.j.shape[1]
            if flux.index_shape != (n_vec,):
                raise ValueError(
                    f"Flux expression has {n_vec} components per direction, but "
                    f"StaggeredGrid {flux.name!r} has index_shape={flux.index_shape}. "
                    f"Expected index_shape=({n_vec},)."
                )
            fluxes = [sp.Matrix(fluxes.tolist()[i]) for i in range(self.dim)]
        else:
            fluxes = [fluxes.tolist()[i] for i in range(self.dim)]

        A0 = sum([sp.Matrix(d).norm() for d in flux.stencil.staggered_entries]) / self.dim

        discrete_fluxes = []
        for neighbor in flux.stencil.staggered_entries[1:]:
            directional_flux = fluxes[0] * int(neighbor[0])
            for i in range(1, self.dim):
                directional_flux += fluxes[i] * int(neighbor[i])
            discrete_flux = sp.simplify(discretize(directional_flux, neighbor))
            free_weights = [
                s
                for s in discrete_flux.atoms(sp.Symbol)
                if s.name.startswith("fvm_free_")
            ]

            if len(free_weights) > 0 and not is_vector:
                discrete_flux = discrete_flux.collect(
                    discrete_flux.atoms(ps.field.Field.Access)
                )
                access_counts = defaultdict(list)
                for values in itertools.product([-1, 0, 1], repeat=len(free_weights)):
                    subs = {
                        free_weight: value
                        for free_weight, value in zip(free_weights, values)
                    }
                    simp = discrete_flux.subs(subs)
                    access_count = len(simp.atoms(ps.field.Field.Access))
                    access_counts[access_count].append(simp)
                best_count = min(access_counts.keys())
                discrete_flux = sum(access_counts[best_count]) / len(
                    access_counts[best_count]
                )
            discrete_fluxes.append(discrete_flux / sp.Matrix(neighbor).norm())

        n_vec = self.j.shape[1] if is_vector else None

        @ps.flow.block
        def discrete_flux_block(let):
            for i in range(1, flux.stencil.Q // 2 + 1):
                val = sp.simplify(discrete_fluxes[i - 1]) / A0
                if n_vec is not None:
                    for k in range(n_vec):
                        let.store[flux[i](k)] = val[k]
                else:
                    let.store[flux[i]] = val
        return discrete_flux_block

    def discrete_source(self):
        """Return a list of assignments for the discrete source term"""

        def discretize(term):
            if isinstance(term, ps.fd.Diff):
                access, direction = get_access_and_direction(term)

                if self.dim == 2:
                    stencil = [
                        "".join(a).replace(" ", "")
                        for a in itertools.product("NS ", "EW ")
                        if "".join(a).strip()
                    ]
                else:
                    stencil = [
                        "".join(a).replace(" ", "")
                        for a in itertools.product("NS ", "EW ", "TB ")
                        if "".join(a).strip()
                    ]
                weights = None
                for stencil in [
                    ["N", "S", "E", "W", "T", "B"][: 2 * self.dim],
                    stencil,
                ]:
                    stencil = [
                        tuple(ps.stencil.direction_string_to_offset(d, self.dim))
                        for d in stencil
                    ]

                    derivation = FD(direction, stencil).get_stencil()
                    if not derivation.accuracy:
                        continue
                    weights = derivation.weights

                    # if the weights are underdefined, we can choose the free symbols to find the sparsest stencil
                    free_weights = set(
                        itertools.chain(*[w.free_symbols for w in weights])
                    )
                    if len(free_weights) > 0:
                        zero_counts = defaultdict(list)
                        for values in itertools.product(
                            [-1, -sp.Rational(1, 2), 0, 1, sp.Rational(1, 2)],
                            repeat=len(free_weights),
                        ):
                            subs = {
                                free_weight: value
                                for free_weight, value in zip(free_weights, values)
                            }
                            weights = [w.subs(subs) for w in derivation.weights]
                            if not all(a == 0 for a in weights):
                                zero_count = sum([1 for w in weights if w == 0])
                                zero_counts[zero_count].append(weights)
                        best = zero_counts[max(zero_counts.keys())]
                        if len(best) > 1:
                            raise NotImplementedError(
                                "more than one suitable set of weights found, "
                                "don't know how to proceed"
                            )
                        weights = best[0]
                    break
                if not weights:
                    raise Exception(
                        "the requested derivative cannot be performed with the available neighbors"
                    )
                assert weights

                if access._field.index_dimensions == 0:
                    return sum(
                        [
                            access._field.__getitem__(point) * weight
                            for point, weight in zip(stencil, weights)
                        ]
                    )
                else:
                    total = (
                        access.get_shifted(*stencil[0]).at_index(*access.index)
                        * weights[0]
                    )
                    for point, weight in zip(stencil[1:], weights[1:]):
                        addl = (
                            access.get_shifted(*point).at_index(*access.index) * weight
                        )
                        total += addl
                    return total

            if term.args:
                new_args = [discretize(a) for a in term.args]
                return term.func(*new_args)
            else:
                return term

        source = self.q.applyfunc(ps.fd.derivative.expand_diff_full)
        source = source.applyfunc(discretize)

        return [
            ps.Assignment(lhs, rhs)
            for lhs, rhs in zip(self.c.center_vector, sp.flatten(source))
            if rhs
        ]

    def discrete_continuity(self, flux: ps.StaggeredField):
        """Return a list of assignments for the continuity equation, which includes the source term

        Args:
            flux: a staggered grid from which the fluxes are taken
        """

        assert isinstance(flux, ps.StaggeredField)

        neighbors = flux.stencil.stencil_entries

        divergence = flux.face[1]
        for d in range(2, len(neighbors)):
            divergence += flux.face[d]

        source = self.discrete_source()
        source = {s.lhs: s.rhs for s in source}
        lhs = self.c.center_vector[0, 0]

        @ps.flow.block
        def discrete_continuity_block(let):
            if lhs in source:
                let.store[lhs] = lhs - divergence + source[lhs]
            else:
                let.store[lhs] = lhs - divergence
        return discrete_continuity_block


def VOF(j: ps.StaggeredField, v: ps.field.Field, ρ: ps.field.Field):
    """Volume-of-fluid discretization of advection

    Args:
        j: the staggered field to write the fluxes to. Should have a D2Q9/D3Q27 stencil. Other stencils work too, but
           incur a small error (D2Q5/D3Q7: v^2, D3Q19: v^3).
        v: the flow velocity field
        ρ: the quantity to advect
    """
    assert isinstance(j, ps.StaggeredField)

    fluxes = [[] for i in range(len(j.stencil.staggered_entries) - 1)]

    v0 = v.center_vector
    for d, neighbor in enumerate(j.stencil.staggered_entries[1:]):
        v1 = v.neighbor_vector(neighbor)

        # going out
        cond = sp.And(*[sp.Or(neighbor[i] * v0[i] > 0, neighbor[i] == 0) for i in range(len(v0))])
        overlap1 = [1 - sp.Abs(v0[i]) for i in range(len(v0))]
        overlap2 = [neighbor[i] * v0[i] for i in range(len(v0))]
        overlap = sp.Mul(*[(overlap1[i] if neighbor[i] == 0 else overlap2[i]) for i in range(len(v0))])
        fluxes[d].append(ρ.center_vector * overlap * sp.Piecewise((1, cond), (0, True)))

        # coming in
        cond = sp.And(*[sp.Or(neighbor[i] * v1[i] < 0, neighbor[i] == 0) for i in range(len(v1))])
        overlap1 = [1 - sp.Abs(v1[i]) for i in range(len(v1))]
        overlap2 = [v1[i] for i in range(len(v1))]
        overlap = sp.Mul(*[(overlap1[i] if neighbor[i] == 0 else overlap2[i]) for i in range(len(v1))])
        sign = sum([1 if n == 1 else 0 for n in neighbor]) % 2 * 2 - 1
        fluxes[d].append(sign * ρ.neighbor_vector(neighbor) * overlap * sp.Piecewise((1, cond), (0, True)))

    for i, ff in enumerate(fluxes):
        fluxes[i] = ff[0]
        for f in ff[1:]:
            fluxes[i] += f

    @ps.flow.block
    def VOF_block(let):
        for i, d in enumerate(j.stencil.staggered_entries[1:]):
            let.store[j[i + 1]] = fluxes[i][0]
    return VOF_block
