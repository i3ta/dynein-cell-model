#pragma once

#include <memory>
#include <string>

#include "dynein_cell_model/state.h"

namespace dynein_cell_model {

class OutputWriter {
public:
  explicit OutputWriter(const std::string &path);
  ~OutputWriter();
  OutputWriter(OutputWriter &&) noexcept;
  OutputWriter &operator=(OutputWriter &&) noexcept;
  OutputWriter(const OutputWriter &) = delete;
  OutputWriter &operator=(const OutputWriter &) = delete;

  void saveState(const CellState &state);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/** Save the current state through its attached OutputWriter. */
void saveState(CellState &state);

/** Attach a new output writer, replacing any existing writer. */
void setOutput(CellState &state, const std::string &path);

} // namespace dynein_cell_model
