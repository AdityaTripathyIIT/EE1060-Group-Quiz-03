package main

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
)

// Parameters
const (
	L     = 1.0
	R     = 1.0
	A     = 1.0
	T     = 1.0
	alpha = 0.5
	N     = 1000   // Coarse time steps
	dt    = 0.01   // Coarse time step size
	K     = 2      // Parareal iterations
	M     = 200    // Number of fine solver substeps per coarse step
	chunkSize = 10 // Work batch size for parallel processing
)

// Square wave function
func V(t float64) float64 {
	modTime := math.Mod(t, T)
	if modTime < alpha*T {
		return A
	}
	return 0
}

// RK4 Step for fine solver
func RK4Step(i float64, t float64, dt float64) float64 {
	f := func(t, i float64) float64 { return (V(t) - R*i) / L }

	k1 := dt * f(t, i)
	k2 := dt * f(t+dt/2, i+k1/2)
	k3 := dt * f(t+dt/2, i+k2/2)
	k4 := dt * f(t+dt, i+k3)

	return i + (k1+2*k2+2*k3+k4)/6
}

// Full RK4 solution
func SolveRK4(filename string) {
	file, _ := os.Create(filename)
	defer file.Close()

	i := 0.0
	t := 0.0
  dt := 1e-4
  N := int(1e5) 
	for step := 0; step < N; step++ {
		fmt.Fprintf(file, "%f %f\n", t, i)
		i = RK4Step(i, t, dt)
		t += dt
	}
}

// Parareal Algorithm
func Parareal(filename string) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	var wg sync.WaitGroup
	var mu sync.Mutex

	pool := sync.Pool{
		New: func() interface{} { return make([]float64, N+1) },
	}
	i0 := pool.Get().([]float64)
	i1 := pool.Get().([]float64)
	t := pool.Get().([]float64)
	defer pool.Put(i0)
	defer pool.Put(i1)
	defer pool.Put(t)

	for step := 0; step <= N; step++ {
		t[step] = float64(step) * dt
	}

	// Initial coarse solution (Euler method)
	for step := 0; step < N; step++ {
		i0[step+1] = i0[step] + dt*(V(t[step])-R*i0[step])/L
	}

	// Parareal Iterations
	for k := 0; k < K; k++ {
		wg.Add(N / chunkSize)
		for step := 0; step < N; step += chunkSize {
			go func(start int) {
				defer wg.Done()
				for j := start; j < start+chunkSize && j < N; j++ {
					fine := i0[j]
					fine_dt := dt / float64(M)
					for m := 0; m < M; m++ {
						fine = RK4Step(fine, t[j]+float64(m)*fine_dt, fine_dt)
					}

					mu.Lock()
					i1[j+1] = fine + i0[j+1] - i0[j] 
					mu.Unlock()
				}
			}(step)
		}
		wg.Wait()
		copy(i0, i1)
	}

	file, _ := os.Create(filename)
	defer file.Close()
	for step := 0; step <= N; step++ {
		fmt.Fprintf(file, "%f %f\n", t[step], i0[step])
	}
}

func main() {
	SolveRK4("rk4.txt")
	Parareal("parareal.txt")
}

