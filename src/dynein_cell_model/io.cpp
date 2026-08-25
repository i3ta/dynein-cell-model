#include "dynein_cell_model/io.h"
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5PropertyList.hpp>
#include <map>
#include <stdexcept>
namespace dynein_cell_model {

namespace {

constexpr size_t chunkSize = 100;

template <typename T>
void append(HighFive::File &file, std::map<std::string, size_t> &next,
            const std::string &name,
            const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor> &mat) {
  if (!file.exist(name)) {
    HighFive::DataSpace space(
        {chunkSize, size_t(mat.rows()), size_t(mat.cols())},
        {HighFive::DataSpace::UNLIMITED, size_t(mat.rows()),
         size_t(mat.cols())});
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking({1, size_t(mat.rows()), size_t(mat.cols())}));
    file.createDataSet<T>(name, space, props);
    next[name] = 0;
  }
  auto dset = file.getDataSet(name);
  size_t index =
      next.try_emplace(name, dset.getSpace().getDimensions()[0]).first->second;
  const auto dims = dset.getSpace().getDimensions();
  if (index >= dims[0])
    dset.resize({dims[0] + chunkSize, dims[1], dims[2]});
  dset.select({index, 0, 0}, {1, size_t(mat.rows()), size_t(mat.cols())})
      .write_raw(mat.data());
  ++next[name];
}

void appendTime(HighFive::File &file, int time) {
  if (!file.exist("t")) {
    HighFive::DataSpace space({0}, {HighFive::DataSpace::UNLIMITED});
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking({1}));
    file.createDataSet<int>("t", space, props);
  }
  auto dset = file.getDataSet("t");
  const size_t index = dset.getSpace().getDimensions()[0];
  dset.resize({index + 1});
  dset.select({index}, {1}).write(&time);
}

void appendMask(HighFive::File &file, std::map<std::string, size_t> &next,
                const std::string &name, const ViewI &mat) {
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      bytes = (mat.array() != 0).cast<unsigned char>();
  append(file, next, name, bytes);
}

void trim(HighFive::File &file, const std::map<std::string, size_t> &next) {
  for (const auto &[name, count] : next) {
    auto dset = file.getDataSet(name);
    const auto dims = dset.getSpace().getDimensions();
    dset.resize({count, dims[1], dims[2]});
  }
}

} // namespace

class OutputWriter::Impl {
public:
  explicit Impl(const std::string &path)
      : file(path, HighFive::File::ReadWrite | HighFive::File::Create |
                       HighFive::File::Truncate) {}
  HighFive::File file;
  std::map<std::string, size_t> next;
};

OutputWriter::OutputWriter(const std::string &path)
    : impl(std::make_unique<Impl>(path)) {}

OutputWriter::~OutputWriter() {
  if (impl)
    trim(impl->file, impl->next);
}

OutputWriter::OutputWriter(OutputWriter &&) noexcept = default;

OutputWriter &OutputWriter::operator=(OutputWriter &&) noexcept = default;

void OutputWriter::saveState(const CellState &state) {
  using FloatMatrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const FloatMatrix a = state.A.cast<float>(), i = state.I.cast<float>(),
                    ac = state.AC.cast<float>(), ic = state.IC.cast<float>(),
                    f = state.F.cast<float>(), k0 = state.k0Adh.cast<float>();
  appendTime(impl->file, state.t);
  appendMask(impl->file, impl->next, "cell", state.cell);
  appendMask(impl->file, impl->next, "nuc", state.nuc);
  append(impl->file, impl->next, "A", a);
  append(impl->file, impl->next, "I", i);
  append(impl->file, impl->next, "AC", ac);
  append(impl->file, impl->next, "IC", ic);
  appendMask(impl->file, impl->next, "adh", ViewI(state.adh));
  append(impl->file, impl->next, "F", f);
  append(impl->file, impl->next, "k0_adh", k0);
}

void setOutput(CellState &state, const std::string &path) {
  if (path.empty())
    throw std::invalid_argument("output path must not be empty");
  state.output = std::make_unique<OutputWriter>(path);
}

void saveState(CellState &state) {
  if (!state.output)
    throw std::runtime_error(
        "No output writer is attached; call setOutput first.");
  state.output->saveState(state);
}

} // namespace dynein_cell_model
