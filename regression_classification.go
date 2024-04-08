package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
)

// Struct untuk menyimpan data
type DataPoint struct {
	X float64
	Y float64
}

// Fungsi untuk membaca dataset dari file CSV
func readCSV(filePath string) ([]DataPoint, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var dataset []DataPoint
	for _, line := range lines {
		x, err := strconv.ParseFloat(line[0], 64)
		if err != nil {
			return nil, err
		}
		y, err := strconv.ParseFloat(line[1], 64)
		if err != nil {
			return nil, err
		}
		dataset = append(dataset, DataPoint{X: x, Y: y})
	}
	return dataset, nil
}

// Fungsi untuk normalisasi dataset
func normalizeDataset(dataset []DataPoint) ([]DataPoint, float64, float64) {
	var minX, minY, maxX, maxY float64
	for _, point := range dataset {
		if point.X < minX {
			minX = point.X
		}
		if point.X > maxX {
			maxX = point.X
		}
		if point.Y < minY {
			minY = point.Y
		}
		if point.Y > maxY {
			maxY = point.Y
		}
	}

	var normalizedDataset []DataPoint
	for _, point := range dataset {
		normalizedX := (point.X - minX) / (maxX - minX)
		normalizedY := (point.Y - minY) / (maxY - minY)
		normalizedDataset = append(normalizedDataset, DataPoint{X: normalizedX, Y: normalizedY})
	}
	return normalizedDataset, maxX, maxY
}

// Fungsi untuk menghitung prediksi menggunakan model regresi linear
func predict(x float64, theta0 float64, theta1 float64) float64 {
	return theta0 + theta1*x
}

// Fungsi untuk melatih model regresi linear menggunakan metode gradien turun
func trainLinearRegression(dataset []DataPoint, alpha float64, iterations int) (float64, float64) {
	theta0 := 0.0
	theta1 := 0.0
	m := float64(len(dataset))

	for i := 0; i < iterations; i++ {
		sumTheta0 := 0.0
		sumTheta1 := 0.0
		for _, point := range dataset {
			prediction := predict(point.X, theta0, theta1)
			sumTheta0 += prediction - point.Y
			sumTheta1 += (prediction - point.Y) * point.X
		}
		theta0 -= (alpha / m) * sumTheta0
		theta1 -= (alpha / m) * sumTheta1
	}

	return theta0, theta1
}

func main() {
	// Membaca dataset dari file CSV
	dataset, err := readCSV("data.csv")
	if err != nil {
		log.Fatal("Error reading dataset:", err)
	}

	// Normalisasi dataset
	normalizedDataset, maxX, maxY := normalizeDataset(dataset)

	// Melatih model regresi linear
	alpha := 0.01 // Learning rate
	iterations := 1000
	theta0, theta1 := trainLinearRegression(normalizedDataset, alpha, iterations)

	// Menampilkan parameter model yang sudah dilatih
	fmt.Println("Model regresi linear yang sudah dilatih:")
	fmt.Printf("Theta0: %.2f, Theta1: %.2f\n", theta0, theta1)

	// Contoh prediksi
	x := 0.5 // Misalkan x yang akan diprediksi
	predictedY := predict(x, theta0, theta1)

	// Denormalisasi prediksi
	predictedY = predictedY*(maxY-maxY) + maxY

	fmt.Printf("Prediksi untuk x=%.2f: %.2f\n", x, predictedY)
}
