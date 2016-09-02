#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mandelbrot2.cuh"

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

const char* image_name = "mandel.ppm";
const char* text_name = "mandel.txt";

int* mandelbrot_gpu(float x0, float y0, float x1, float y1, int width, int height, float iterations_limit)
{
	int *d_mandelbrot, *h_mandelbrot;

	CUDA_CHECK_RETURN(cudaMalloc(&d_mandelbrot, sizeof(int) * width * height));
	CUDA_CHECK_RETURN(cudaMemset(d_mandelbrot, 42, sizeof(int) * width * height));

	mandelbrot2(d_mandelbrot, width, height, iterations_limit, x0, y0, x1, y1);

	CUDA_CHECK_RETURN(cudaMallocHost(&h_mandelbrot, sizeof(int) * width * height));
	CUDA_CHECK_RETURN(cudaMemcpy(h_mandelbrot, d_mandelbrot, sizeof(int) * width * height, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_mandelbrot));

	return h_mandelbrot;
}

void makePicture(int *mandel, int width, int height, int MAX)
{

	int red_value, green_value, blue_value;
	red_value = green_value = blue_value = 0;

	int MyPalette[41][3] =
	{
	{ 255, 255, 255 }, //0
			{ 255, 255, 255 }, //1 not used
			{ 255, 255, 255 }, //2 not used
			{ 255, 255, 255 }, //3 not used
			{ 255, 255, 255 }, //4 not used
			{ 255, 180, 255 }, //5
			{ 255, 180, 255 }, //6 not used
			{ 255, 180, 255 }, //7 not used
			{ 248, 128, 240 }, //8
			{ 248, 128, 240 }, //9 not used
			{ 240, 64, 224 }, //10
			{ 240, 64, 224 }, //11 not used
			{ 232, 32, 208 }, //12
			{ 224, 16, 192 }, //13
			{ 216, 8, 176 }, //14
			{ 208, 4, 160 }, //15
			{ 200, 2, 144 }, //16
			{ 192, 1, 128 }, //17
			{ 184, 0, 112 }, //18
			{ 176, 0, 96 }, //19
			{ 168, 0, 80 }, //20
			{ 160, 0, 64 }, //21
			{ 152, 0, 48 }, //22
			{ 144, 0, 32 }, //23
			{ 136, 0, 16 }, //24
			{ 128, 0, 0 }, //25
			{ 120, 16, 0 }, //26
			{ 112, 32, 0 }, //27
			{ 104, 48, 0 }, //28
			{ 96, 64, 0 }, //29
			{ 88, 80, 0 }, //30
			{ 80, 96, 0 }, //31
			{ 72, 112, 0 }, //32
			{ 64, 128, 0 }, //33
			{ 56, 144, 0 }, //34
			{ 48, 160, 0 }, //35
			{ 40, 176, 0 }, //36
			{ 32, 192, 0 }, //37
			{ 16, 224, 0 }, //38
			{ 8, 240, 0 }, //39
			{ 0, 0, 0 } //40
	};

	FILE *f = fopen(image_name, "wb");
	//FILE *ind = fopen(index_name, "w");
	fprintf(f, "P6\n%i %i 255\n", width, height);
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			 int indx = (int) floor(5.0 * log2f(1.0f * mandel[j * width + i] + 1));
			 //fprintf(ind, "%d ", indx);

			 red_value = MyPalette[indx][0];
			 green_value = MyPalette[indx][2];
			 blue_value = MyPalette[indx][1];

			 /*
			red_value = mandel[j * width + i] / (float) MAX * 255;
			green_value = mandel[j * width + i] / (float) MAX * 255;
			blue_value = mandel[j * width + i] / (float) MAX * 255;
			 */
			fputc(red_value, f);   // 0 .. 255
			fputc(green_value, f); // 0 .. 255
			fputc(blue_value, f);  // 0 .. 255
		}
		//fprintf(ind, "\n");
	}
	fclose(f);

}

void dumpToFile(int* mandelbrot, int width, int height)
{
	FILE *f = fopen(text_name, "w");
	for(int y = 0; y < height; y++)
	{
		for(int x = 0; x < width; x++)
			fprintf(f, "%6d ", mandelbrot[y * width + x]);
		fprintf(f, "\n");
	}
	fclose(f);
}

int main(void)
{
	const float x0 = -2;
	const float y0 = -1.314;
	const float x1 = 1;
	const float y1 = 1.314;

	const int sizeX = 10000;
	const int sizeY = sizeX * (y1 - y0) / (x1 - x0);
	const int iters = 1024;

	int* mandelbrot = mandelbrot_gpu(x0, y0, x1, y1, sizeX, sizeY, iters);
	makePicture(mandelbrot, sizeX, sizeY, iters);
	//dumpToFile(mandelbrot, sizeX, sizeY);

	cudaFreeHost(mandelbrot);

	return 0;
}


static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

