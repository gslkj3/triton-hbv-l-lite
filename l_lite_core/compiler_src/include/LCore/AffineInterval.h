#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::triton::lcore {
// Interval algebra is not a legality certificate: callers also prove SSA
// meaning, actual bit-width validity and memory independence.
struct AffinePidFootprint {
  int64_t pidStride = 0;
  int64_t localMin = 0;
  int64_t localMax = 0;
};

inline std::optional<int64_t> checkedI64(__int128 value) {
  if (value < std::numeric_limits<int64_t>::min() ||
      value > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(value);
}

inline std::optional<AffinePidFootprint>
combineFootprints(const AffinePidFootprint &lhs,
                  const AffinePidFootprint &rhs, bool subtract) {
  auto stride = checkedI64(static_cast<__int128>(lhs.pidStride) +
                           (subtract ? -static_cast<__int128>(rhs.pidStride)
                                     : static_cast<__int128>(rhs.pidStride)));
  auto minimum = checkedI64(static_cast<__int128>(lhs.localMin) +
                            (subtract ? -static_cast<__int128>(rhs.localMax)
                                      : static_cast<__int128>(rhs.localMin)));
  auto maximum = checkedI64(static_cast<__int128>(lhs.localMax) +
                            (subtract ? -static_cast<__int128>(rhs.localMin)
                                      : static_cast<__int128>(rhs.localMax)));
  if (!stride || !minimum || !maximum)
    return std::nullopt;
  return AffinePidFootprint{*stride, *minimum, *maximum};
}

inline std::optional<AffinePidFootprint>
scaleFootprint(const AffinePidFootprint &input, int64_t scale) {
  auto stride = checkedI64(static_cast<__int128>(input.pidStride) * scale);
  auto first = checkedI64(static_cast<__int128>(input.localMin) * scale);
  auto second = checkedI64(static_cast<__int128>(input.localMax) * scale);
  if (!stride || !first || !second)
    return std::nullopt;
  return AffinePidFootprint{*stride, std::min(*first, *second),
                            std::max(*first, *second)};
}
} // namespace mlir::triton::lcore
