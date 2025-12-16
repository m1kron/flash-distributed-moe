

#include "gtest/gtest.h"
#include "src/tools/compileTimeSelector.h"

struct params {
  constexpr static int N = 2;
};

template <typename PARAMS>
struct Bla1 {
  static constexpr bool Supports() { return true; }

  int type() { return 1; }
};

template <typename PARAMS>
struct Bla2 {
  static constexpr bool Supports() { return PARAMS::N == 1; }

  int type() { return 2; }
};

template <typename PARAMS>
struct Bla3 {
  static constexpr bool Supports() { return PARAMS::N == 2; }

  int type() { return 3; }
};

TEST(moeTools, compileTimeSelectror_basic) {
  using T1 = moe::tools::CompileTimeSelector<Bla2<params>, Bla3<params>,
                                             Bla1<params>>::type;
  T1 t1;
  ASSERT_EQ(t1.type(), 3);

  using T2 = moe::tools::CompileTimeSelector<Bla2<params>, Bla1<params>>::type;
  T2 t2;
  ASSERT_EQ(t2.type(), 1);
}