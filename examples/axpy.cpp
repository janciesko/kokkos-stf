//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief An AXPY kernel implemented with CUDA kernel in a task
 *
 */

#include <cuda/experimental/stf.cuh>
#include <Kokkos_Core.hpp>

using namespace cuda::experimental::stf;
typedef Kokkos::TeamPolicy<Kokkos::ExecutionSpace>::member_type member_type;

template <typename T>
struct axpy{
  T a;
  Kokkos::View<const T> x;
  Kokkos::View<T> y;

  axpy(T a_, Kokkos::View<const T> x_, Kokkos::View<T> y_)):a(a_),x(x_),y(y_){}
  
  KOKKOS_FUNCTION void operator(member_type team)(){
    T row_val = 0;
    i = team.league_rank();
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,x.extent(1))), [=](int i, int & row_val_local)
    {
     row_val_local += a * x(j);
    }, row_val
    );
    team_member.team_barrier();
    y(i) = row_val;
  }    
}

void axpy_launch(auto s, double alpha, auto x_, auto y_)
{
  auto x (x_);
  auto y (y_);
  Kokkos::parallel_for("spmv",Kokkos::TeamPolicy<> (x.extent(), 16),axpy{alpha,x,y});
}

double X0(int i)
{
  return sin(std::static_cast<double>(i));
}

double Y0(int i)
{
  return cos(std::static_cast<double>(i));
}

int main()
{
  Kokkos::initialize();
  context ctx;
  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  double alpha = 3.14;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  /* Compute Y = Y + alpha X */
  ctx.task(lX.read(), lY.rw())->*[&](cudaStream_t s, auto dX, auto dY) {
    axpy_launch(s, alpha, dx, dy);
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }

  Kokkos::finalize();
}
