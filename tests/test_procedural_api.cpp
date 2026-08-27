#include <filesystem>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include "dynein_cell_model/dynein_cell_model.h"
#include "dynein_cell_model/diffusion.h"
#include "dynein_cell_model/io.h"
#include "dynein_cell_model/simulate.h"

namespace dcm = dynein_cell_model;

namespace {
dcm::CellModelConfig smallConfig() {
  dcm::CellModelConfig config;
  config.simRows = 20;
  config.simCols = 20;
  config.adhNum = 1;
  config.framePadding = 2;
  config.seed = 42;
  config.saveT = 1;
  config.adhT = 100;
  config.frT = 100;
  return config;
}

dcm::ViewI disk(int rows, int cols, int radius) {
  dcm::ViewI result = dcm::ViewI::Zero(rows, cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      if ((r - rows / 2) * (r - rows / 2) + (c - cols / 2) * (c - cols / 2) <=
          radius * radius)
        result(r, c) = 1;
  return result;
}

dcm::ViewMask environment(int rows, int cols) {
  dcm::ViewMask result(rows, cols);
  std::vector<Eigen::Triplet<int>> entries;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      entries.emplace_back(r, c, 1);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
} // namespace

TEST(ProceduralApi, ConfigRoundTripsEverySavedField) {
  auto config = smallConfig();
  config.g = 1.7;
  config.propFactor = 0.4;
  config.saveDir = "snapshots";
  config.diffusionBackend = "kokkos";
  config.t = 3;
  const auto path =
      std::filesystem::temp_directory_path() / "dcm-config-roundtrip.yaml";
  config.saveFile(path.string());
  const dcm::CellModelConfig loaded(path.string());
  EXPECT_EQ(loaded.simRows, config.simRows);
  EXPECT_EQ(loaded.adhNum, config.adhNum);
  EXPECT_DOUBLE_EQ(loaded.g, config.g);
  EXPECT_DOUBLE_EQ(loaded.propFactor, config.propFactor);
  EXPECT_EQ(loaded.saveDir, config.saveDir);
  EXPECT_EQ(loaded.diffusionBackend, config.diffusionBackend);
  EXPECT_EQ(loaded.t, config.t);
  std::filesystem::remove(path);
}

TEST(ProceduralApi, KokkosDiffusionMatchesEigenReference) {
#ifndef DCM_ENABLE_KOKKOS
  GTEST_SKIP() << "Kokkos support was disabled at build time";
#else
  auto config = smallConfig();
  config.diffT = 3;
  const auto cell = disk(20, 20, 6);
  const auto nuc = disk(20, 20, 2);
  const auto env = environment(20, 20);
  auto eigenState = dcm::initializeState(config, cell, nuc, env);
  auto kokkosState = dcm::initializeState(config, cell, nuc, env);

  auto initializeFields = [](dcm::CellState &state) {
    for (int row = 0; row < state.config.simRows; ++row) {
      for (int col = 0; col < state.config.simCols; ++col) {
        const double value = 0.01 * row + 0.001 * col;
        state.k0Adh(row, col) = 0.02 + 0.0001 * value;
        state.A(row, col) = 0.15 + value;
        state.I(row, col) = 0.70 - 0.1 * value;
        state.F(row, col) = 0.05 + 0.2 * value;
        state.AC(row, col) = 0.25 + 0.5 * value;
        state.IC(row, col) = 0.60 - 0.15 * value;
        state.FC(row, col) = 0.10 + 0.1 * value;
      }
    }
  };
  initializeFields(eigenState);
  initializeFields(kokkosState);

  dcm::diffuseK0AdhEigen(eigenState);
  dcm::diffuseK0AdhKokkos(kokkosState);

  EXPECT_TRUE(eigenState.A.isApprox(kokkosState.A, 1e-12));
  EXPECT_TRUE(eigenState.I.isApprox(kokkosState.I, 1e-12));
  EXPECT_TRUE(eigenState.F.isApprox(kokkosState.F, 1e-12));
  EXPECT_TRUE(eigenState.AC.isApprox(kokkosState.AC, 1e-12));
  EXPECT_TRUE(eigenState.IC.isApprox(kokkosState.IC, 1e-12));
  EXPECT_TRUE(eigenState.FC.isApprox(kokkosState.FC, 1e-12));
#endif
}

TEST(ProceduralApi, InitializesAndValidatesMasksDeterministically) {
  const auto config = smallConfig();
  const auto cell = disk(20, 20, 6), nuc = disk(20, 20, 2);
  const auto env = environment(20, 20);
  auto first = dcm::initializeState(config, cell, nuc, env);
  auto second = dcm::initializeState(config, cell, nuc, env);
  EXPECT_EQ(first.V0, cell.sum());
  EXPECT_EQ(first.V0Nuc, nuc.sum());
  EXPECT_EQ(first.rng(), second.rng());
  EXPECT_THROW(dcm::initializeState(config, dcm::ViewI::Zero(3, 3), nuc, env),
               std::invalid_argument);
}

TEST(ProceduralApi, OutputIsOptionalAndWriterFinalizesDatasets) {
  const auto config = smallConfig();
  const auto cell = disk(20, 20, 6);
  const auto nuc = disk(20, 20, 2);
  const auto env = environment(20, 20);
  auto state = dcm::initializeState(config, cell, nuc, env);
  EXPECT_NO_THROW(dcm::simulateSteps(state, 1));
  EXPECT_THROW(dcm::saveState(state), std::runtime_error);

  const auto path =
      std::filesystem::temp_directory_path() / "dcm-procedural-output.h5";
  {
    auto written = dcm::initializeState(config, cell, nuc, env);
    dcm::setOutput(written, path.string());
    dcm::saveState(written);
    dcm::simulateSteps(written, 1);
  }
  HighFive::File file(path.string(), HighFive::File::ReadOnly);
  EXPECT_EQ(file.getDataSet("t").getSpace().getDimensions()[0], 2);
  EXPECT_EQ(file.getDataSet("cell").getSpace().getDimensions()[0], 2);
  std::filesystem::remove(path);
}
