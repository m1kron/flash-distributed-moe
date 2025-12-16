#pragma once
#include <type_traits>

namespace moe {
namespace tools {

// Compile time selector select the first Candidate, for which Supports returns
// true.
template <typename... Candidates>
struct CompileTimeSelector;

template <typename Last>
struct CompileTimeSelector<Last> {
  static constexpr bool match = Last::Supports();
  using type = std::conditional_t<match, Last, void>;
};

template <typename First, typename... Rest>
struct CompileTimeSelector<First, Rest...> {
  static constexpr bool match = First::Supports();
  using type = std::conditional_t<match, First,
                                  typename CompileTimeSelector<Rest...>::type>;
};

}  // namespace tools
}  // namespace moe