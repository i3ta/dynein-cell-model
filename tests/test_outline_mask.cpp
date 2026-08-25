#include <gtest/gtest.h>

#include "dynein_cell_model/mask.h"

namespace dcm = dynein_cell_model;

TEST(OutlineMask, TracksDenseMembershipAndCoordinates) {
  dcm::OutlineMask outline(4, 3);
  outline.set(2, 1);
  outline.set(0, 2);
  outline.set(2, 1);

  EXPECT_TRUE(outline.contains(2, 1));
  EXPECT_FALSE(outline(1, 1));
  ASSERT_EQ(outline.size(), 2);
  EXPECT_EQ(outline.coords()[0], dcm::OutlineMask::Coord(2, 1));

  outline.unset(2, 1);
  EXPECT_FALSE(outline.contains(2, 1));
  EXPECT_EQ(outline.size(), 1);
}

TEST(OutlineMask, RebuildsAndShufflesWithoutChangingStableOrder) {
  dcm::OutlineMask outline(3, 3);
  outline.set(2, 2);
  outline.set(0, 1);
  outline.set(1, 0);
  outline.rebuildCoordinatesColumnMajor();
  EXPECT_EQ(outline.coords(), (std::vector<dcm::OutlineMask::Coord>{{1, 0}, {0, 1}, {2, 2}}));

  const auto stable = outline.coords();
  std::mt19937 rng(42);
  const auto randomized = outline.shuffled(rng);
  EXPECT_EQ(outline.coords(), stable);
  EXPECT_EQ(randomized.size(), stable.size());
}
