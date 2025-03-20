package main

import "testing"

func BenchmarkParareal(b *testing.B){
  for i := 0; i < b.N; i++{
    Parareal("test.txt")
  }
}

func BenchmarkRK4(b *testing.B){
  for i := 0; i < b.N; i++{
    SolveRK4("test.txt")
  }
}
