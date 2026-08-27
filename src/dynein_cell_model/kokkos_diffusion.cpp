#include "dynein_cell_model/diffusion.h"

#include <stdexcept>

#ifdef DCM_ENABLE_KOKKOS

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>

namespace dynein_cell_model {
namespace {

void ensureKokkosInitialized() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
      std::atexit([] {
        if (Kokkos::is_initialized() && !Kokkos::is_finalized())
          Kokkos::finalize();
      });
    }
  });
}

using DeviceViewD = Kokkos::View<double **, Kokkos::LayoutRight>;
using DeviceViewI = Kokkos::View<int **, Kokkos::LayoutRight>;

template <typename DeviceView, typename EigenMatrix>
DeviceView copyToDevice(const char *label, const EigenMatrix &source) {
  DeviceView device(std::string(label), source.rows(), source.cols());
  auto host = Kokkos::create_mirror_view(device);
  std::copy(source.data(), source.data() + source.size(), host.data());
  Kokkos::deep_copy(device, host);
  return device;
}

template <typename EigenMatrix>
void copyToEigen(const DeviceViewD &source, EigenMatrix &destination) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), source);
  std::copy(host.data(), host.data() + destination.size(), destination.data());
}

} // namespace

void diffuseK0AdhKokkos(CellState &state) {
  ensureKokkosInitialized();
  const auto &config = state.config;

  auto cell = copyToDevice<DeviceViewI>("cell", state.cell);
  auto nuc = copyToDevice<DeviceViewI>("nuc", state.nuc);
  auto k0Adh = copyToDevice<DeviceViewD>("k0_adh", state.k0Adh);
  auto a = copyToDevice<DeviceViewD>("A", state.A);
  auto i = copyToDevice<DeviceViewD>("I", state.I);
  auto f = copyToDevice<DeviceViewD>("F", state.F);
  auto ac = copyToDevice<DeviceViewD>("AC", state.AC);
  auto ic = copyToDevice<DeviceViewD>("IC", state.IC);
  auto fc = copyToDevice<DeviceViewD>("FC", state.FC);
  auto aNew = copyToDevice<DeviceViewD>("A_new", state.A);
  auto iNew = copyToDevice<DeviceViewD>("I_new", state.I);
  auto fNew = copyToDevice<DeviceViewD>("F_new", state.F);
  auto acNew = copyToDevice<DeviceViewD>("AC_new", state.AC);
  auto icNew = copyToDevice<DeviceViewD>("IC_new", state.IC);
  auto fcNew = copyToDevice<DeviceViewD>("FC_new", state.FC);

  const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
      {state.frameRowStart, state.frameColStart},
      {state.frameRowEnd + 1, state.frameColEnd + 1});
  const double s2C = 0.05;
  const DiffusionParams params = state.params;
  const double a0Cubed = std::pow(params.A0, 3);
  const double dx2 = params.dx * params.dx;

  for (int step = 0; step < config.diffT; ++step) {
    Kokkos::parallel_for(
        "dcm_diffuse_k0_adh", policy,
        KOKKOS_LAMBDA(const int row, const int col) {
          if (cell(row, col) != 1)
            return;

          const double a3 = Kokkos::pow(a(row, col), 3);
          const double reaction =
              (k0Adh(row, col) + params.gamma * a3 / (a0Cubed + a3)) *
                  i(row, col) -
              params.delta *
                  (params.s1 +
                   params.s2 * f(row, col) / (params.F0 + f(row, col))) *
                  a(row, col);
          const double h =
              params.eps * (params.kn * a(row, col) - params.ks * f(row, col));

          aNew(row, col) =
              a(row, col) +
              params.dt *
                  (reaction +
                   params.DA / dx2 *
                       (cell(row + 1, col) * (a(row + 1, col) - a(row, col)) -
                        cell(row - 1, col) * (a(row, col) - a(row - 1, col)) +
                        cell(row, col + 1) * (a(row, col + 1) - a(row, col)) -
                        cell(row, col - 1) * (a(row, col) - a(row, col - 1))));
          iNew(row, col) =
              i(row, col) +
              params.dt *
                  (-reaction +
                   params.DI / dx2 *
                       (cell(row + 1, col) * (i(row + 1, col) - i(row, col)) -
                        cell(row - 1, col) * (i(row, col) - i(row - 1, col)) +
                        cell(row, col + 1) * (i(row, col + 1) - i(row, col)) -
                        cell(row, col - 1) * (i(row, col) - i(row, col - 1))));
          fNew(row, col) = f(row, col) + h * params.dt;

          if (nuc(row, col) == 0) {
            const double ac3 = Kokkos::pow(ac(row, col), 3);
            const double reactionC =
                (params.k0 + params.gamma * ac3 / (a0Cubed + ac3)) *
                    ic(row, col) -
                params.delta *
                    (params.s1 +
                     s2C * fc(row, col) / (params.F0 + fc(row, col))) *
                    ac(row, col);
            const double hC = params.eps * (params.kn * ac(row, col) -
                                            params.ks * fc(row, col));
            acNew(row, col) =
                ac(row, col) +
                params.dt *
                    (reactionC + params.DA / dx2 *
                                     ((cell(row + 1, col) - nuc(row + 1, col)) *
                                          (ac(row + 1, col) - ac(row, col)) -
                                      (cell(row - 1, col) - nuc(row - 1, col)) *
                                          (ac(row, col) - ac(row - 1, col)) +
                                      (cell(row, col + 1) - nuc(row, col + 1)) *
                                          (ac(row, col + 1) - ac(row, col)) -
                                      (cell(row, col - 1) - nuc(row, col - 1)) *
                                          (ac(row, col) - ac(row, col - 1))));
            icNew(row, col) =
                ic(row, col) +
                params.dt * (-reactionC +
                             params.DI / dx2 *
                                 ((cell(row + 1, col) - nuc(row + 1, col)) *
                                      (ic(row + 1, col) - ic(row, col)) -
                                  (cell(row - 1, col) - nuc(row - 1, col)) *
                                      (ic(row, col) - ic(row - 1, col)) +
                                  (cell(row, col + 1) - nuc(row, col + 1)) *
                                      (ic(row, col + 1) - ic(row, col)) -
                                  (cell(row, col - 1) - nuc(row, col - 1)) *
                                      (ic(row, col) - ic(row, col - 1))));
            fcNew(row, col) = fc(row, col) + hC * params.dt;
          }
        });
    std::swap(a, aNew);
    std::swap(i, iNew);
    std::swap(f, fNew);
    std::swap(ac, acNew);
    std::swap(ic, icNew);
    std::swap(fc, fcNew);
  }

  Kokkos::fence();
  copyToEigen(a, state.A);
  copyToEigen(i, state.I);
  copyToEigen(f, state.F);
  copyToEigen(ac, state.AC);
  copyToEigen(ic, state.IC);
  copyToEigen(fc, state.FC);
}

} // namespace dynein_cell_model

#else

namespace dynein_cell_model {
void diffuseK0AdhKokkos(CellState &) {
  throw std::runtime_error(
      "Kokkos diffusion support was disabled at build time");
}
} // namespace dynein_cell_model

#endif
