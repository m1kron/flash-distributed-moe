#pragma once
#include <type_traits>

namespace moe {
namespace tools {

// Compile time selector select the first Candidate, for which
// AreAllConstraintsSatisfied() returns true.
template <typename... Candidates>
struct CompileTimeSelector;

template <typename Last>
struct CompileTimeSelector<Last> {
  static constexpr bool match = Last::AreAllConstraintsSatisfied();
  using type = std::conditional_t<match, Last, void>;

  static_assert(
      !std::is_same_v<type, void>,
      "There is no candidate, for which all constraints are satisfied.");
};

template <typename First, typename... Rest>
struct CompileTimeSelector<First, Rest...> {
  static constexpr bool match = First::AreAllConstraintsSatisfied();
  using type = std::conditional_t<match, First,
                                  typename CompileTimeSelector<Rest...>::type>;
};

}  // namespace tools
}  // namespace moe