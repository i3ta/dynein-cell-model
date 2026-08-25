#include <fstream>

#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "dynein_cell_model/config.h"
#include "dynein_cell_model/state.h"
#include "dynein_cell_model/types.h"

namespace dynein_cell_model {

CellModelConfig::CellModelConfig() {
  // Conservative defaults keep a directly constructed config fully valid.
  k = 1.6; kNuc = 4; g = 1; T = 500; TNuc = 1; actSlope = 0.03;
  adhSigma = 5; adhBasal = 0.3; adhFrac = 0.03; adhNum = 50; R0 = 20;
  RNuc = 1; dynBasal = 0.9; propFactor = 1; dynSigma = 8.56; dynScale = 0.683;
  DA = 0.0003333333; DI = 0.0333333333; k0Min = 0.01; gamma = 1;
  delta = 1; A0 = 0.4; F0 = 0.5; kn = 1; ks = 0.25; eps = 0.1;
  AMax = 1; AMin = 0; ACMax = 1; ACMin = 0; t = 0; saveDir.clear();
  k0 = 0.10;
  k0Scalar = 10;
  k = 1.6;
  T = 500;
  kNuc = 4;
  TNuc = 1;
  RNuc = 1;
  R0 = 20;
  dynBasal = 0.9;
  dynSigma = 8.56;
  dynScale = 0.683;
  s1 = 0.7;
  s2 = 0.2;
  diffT = 100;
  dt = 3.75e-4;
  dx = 7.0755e-3;
  actSlope = 0.03;
  adhNum = 50;
  adhFrac = 0.03;
  adhSigma = 5;
  framePadding = 20;
  saveT = 1000;
  adhT = 200;
  frT = 50;
  adhBasal = 0.3;
  simRows = 1500;
  simCols = 600;
  seed = 0;
  numIters = 5000000;
}

CellModelConfig::CellModelConfig(std::string config_file) : CellModelConfig() {
  // Load file
  YAML::Node config = YAML::LoadFile(config_file);

  // Read config
  k = config["k"].as<double>();
  kNuc = config["k_nuc"].as<double>();
  g = config["g"].as<double>();
  T = config["T"].as<double>();
  TNuc = config["T_nuc"].as<double>();
  actSlope = config["act_slope"].as<double>();
  adhSigma = config["adh_sigma"].as<double>();
  adhBasal = config["adh_basal"].as<double>();
  adhFrac = config["adh_frac"].as<double>();
  adhNum = config["adh_num"].as<int>();
  R0 = config["R0"].as<int>();
  RNuc = config["R_nuc"].as<double>();
  dynBasal = config["dyn_basal"].as<double>();
  propFactor = config["prop_factor"].as<double>();
  dynSigma = config["dyn_sigma"].as<double>();
  dynScale = config["dyn_scale"].as<double>();
  DA = config["DA"].as<double>();
  DI = config["DI"].as<double>();
  k0 = config["k0"].as<double>();
  k0Min = config["k0_min"].as<double>();
  k0Scalar = config["k0_scalar"].as<double>();
  gamma = config["gamma"].as<double>();
  delta = config["delta"].as<double>();
  A0 = config["A0"].as<double>();
  s1 = config["s1"].as<double>();
  s2 = config["s2"].as<double>();
  F0 = config["F0"].as<double>();
  kn = config["kn"].as<double>();
  ks = config["ks"].as<double>();
  eps = config["eps"].as<double>();
  dt = config["dt"].as<double>();
  dx = config["dx"].as<double>();
  AMax = config["A_max"].as<double>();
  AMin = config["A_min"].as<double>();
  ACMax = config["AC_max"].as<double>();
  ACMin = config["AC_min"].as<double>();
  simRows = config["sim_rows"].as<int>();
  simCols = config["sim_cols"].as<int>();
  seed = config["seed"].as<int>();
  numIters = config["num_iters"].as<int>();
  framePadding = config["frame_padding"].as<int>();
  diffT = config["diff_t"].as<int>();
  saveT = config["save_t"].as<int>();
  adhT = config["adh_t"].as<int>();
  frT = config["fr_t"].as<int>();
  t = config["t"] ? config["t"].as<int>() : 0;
  saveDir = config["save_dir"] ? config["save_dir"].as<std::string>() : "";
}

void CellModelConfig::saveFile(std::string dest_file) const {
  YAML::Node config;
  config["k"] = k;
  config["k_nuc"] = kNuc;
  config["g"] = g;
  config["T"] = T;
  config["T_nuc"] = TNuc;
  config["act_slope"] = actSlope;
  config["adh_sigma"] = adhSigma;
  config["adh_basal"] = adhBasal;
  config["adh_frac"] = adhFrac;
  config["adh_num"] = adhNum;
  config["R0"] = R0;
  config["R_nuc"] = RNuc;
  config["dyn_basal"] = dynBasal;
  config["prop_factor"] = propFactor;
  config["dyn_sigma"] = dynSigma;
  config["dyn_scale"] = dynScale;
  config["DA"] = DA;
  config["DI"] = DI;
  config["k0"] = k0;
  config["k0_min"] = k0Min;
  config["k0_scalar"] = k0Scalar;
  config["gamma"] = gamma;
  config["delta"] = delta;
  config["A0"] = A0;
  config["s1"] = s1;
  config["s2"] = s2;
  config["F0"] = F0;
  config["kn"] = kn;
  config["ks"] = ks;
  config["eps"] = eps;
  config["dt"] = dt;
  config["dx"] = dx;
  config["A_max"] = AMax;
  config["A_min"] = AMin;
  config["AC_max"] = ACMax;
  config["AC_min"] = ACMin;
  config["sim_rows"] = simRows;
  config["sim_cols"] = simCols;
  config["seed"] = seed;
  config["num_iters"] = numIters;
  config["frame_padding"] = framePadding;
  config["diff_t"] = diffT;
  config["save_t"] = saveT;
  config["adh_t"] = adhT;
  config["fr_t"] = frT;
  config["t"] = t;
  config["save_dir"] = saveDir;

  std::ofstream fout(dest_file);
  fout << config;
}

DiffusionParams CellModelConfig::getDiffusionParams() const {
  return DiffusionParams{
      .DA = DA,
      .DI = DI,
      .k0 = k0,
      .k0Min = k0Min,
      .k0Scalar = k0Scalar,
      .gamma = gamma,
      .delta = delta,
      .A0 = A0,
      .s1 = s1,
      .s2 = s2,
      .F0 = F0,
      .kn = kn,
      .ks = ks,
      .eps = eps,
      .dt = dt,
      .dx = dx,
  };
};

} // namespace dynein_cell_model
