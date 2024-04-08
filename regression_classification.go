package main

import (
	"fmt"
	"math/rand"
	"time"
)

// Struct untuk menyimpan data
type DataPoint struct {
	X float64
	Y float64
}

// Fungsi untuk membuat dataset secara acak
func generateDataset(numPoints int) []DataPoint {
	rand.Seed(time.Now().UnixNano())
	var dataset []DataPoint
	for i := 0; i < numPoints; i++ {
		x := rand.Float64() * 10
		y := 2*x + 3 + rand.Float64()*3 // regresi linear y = 2x + 3 + noise
		dataset = append(dataset, DataPoint{X: x, Y: y})
	}
	return dataset
}

// Fungsi untuk melatih model regresi linear
func trainLinearRegression(dataset []DataPoint) (float64, float64) {
	var sumX, sumY, sumXX, sumXY float64
	for _, point := range dataset {
		sumX += point.X
		sumY += point.Y
		sumXX += point.X * point.X
		sumXY += point.X * point.Y
	}
	n := float64(len(dataset))
	b := (n*sumXY - sumX*sumY) / (n*sumXX - sumX*sumX)
	a := (sumY - b*sumX) / n
	return a, b
}

func main() {
	// Membuat dataset dengan 100 titik data
	dataset := generateDataset(100)

	// Melatih model regresi linear
	a, b := trainLinearRegression(dataset)

	// Menampilkan parameter model
	fmt.Printf("Model regresi linear: y = %.2f * x + %.2f\n", b, a)
}
