#ifndef MANDELBROT_CUH_
#define MANDELBROT_CUH_

//INIT_FRAME must be multiply of 4 (vector writes)
//max is 256, min = 16
#define INIT_FRAME 256
#define MAX_THREADS 1024

#define LINEAR
#define LINEAR_STEP 32
__global__ void mandelbrot1(int* iters, int off_x, int off_y, int width, int height, float x0, float y0, float dX, float dY, int iterations_limit)
{
	int stepsDone, stepsToDo, prevStepsToDo;
	int thrX = (blockDim.x * blockIdx.x + threadIdx.x) + off_x;
	int thrY = (blockDim.y * blockIdx.y + threadIdx.y) + off_y;
	float x, y, Zx, Zy, tZx, lZx, lZy;
	if(thrX > width || thrY > height)
		return;

	x = x0 + thrX * dX;
	y = y0 + thrY * dY;
	lZx = Zx = x;
	lZy = Zy = y;
	stepsDone = 0;
	stepsToDo = 0; //absolute number of iterations
	prevStepsToDo = 0; //sure?

	while(((Zx * Zx + Zy * Zy) < 4) && stepsDone < iterations_limit)
	{
		lZx = Zx; //save last Zx and Zy
		lZy = Zy;
		prevStepsToDo = stepsToDo;
#ifdef EXPONENTIAL
		stepsToDo = stepsToDo == 0 ? 1 : stepsToDo <<= 1; //exponential step
#endif

#ifdef LINEAR
		stepsToDo += LINEAR_STEP; //linear step
#endif
		if(stepsToDo > iterations_limit)
			break; //iterations between lower pot and iterations_limit will be done in second loop
		while(stepsDone < stepsToDo)
		{
			tZx = Zx * Zx - Zy * Zy + x;
			Zy = 2 * Zx * Zy + y;
			Zx = tZx;
			stepsDone++;
		}
	}

	Zx = lZx;
	Zy = lZy;
	stepsDone = prevStepsToDo;

	while(((Zx * Zx + Zy * Zy) < 4) && stepsDone < iterations_limit)
	{
		tZx = Zx * Zx - Zy * Zy + x;
		Zy = 2 * Zx * Zy + y;
		Zx = tZx;
		stepsDone++;
	}
	iters[thrY * width + thrX] = stepsDone;
}

__forceinline__ __device__ int compute_point(float x, float y, int iterations_limit)
{
	int i;
	float Zx, Zy, tZx;

	Zx = x;
	Zy = y;
	i = 0;
	while((i < iterations_limit) && ((Zx * Zx + Zy * Zy) < 4))
	{
		tZx = Zx * Zx - Zy * Zy + x;
		Zy = 2 * Zx * Zy + y;
		Zx = tZx;
		i++;
	}

	return i;

}

__global__ void fill_kernel2(int *Mandel, int value, int off_x, int off_y, int width, int height, int frame_height)
{
	int4 val4;
	val4.w = val4.x = val4.y = val4.z = value;
	int x = off_x + 4 * threadIdx.x;
	int y = off_y + threadIdx.y;

	for(; y < off_y + frame_height; y += blockDim.y)
		if(x < width && y < height)
			*((int4*) (Mandel + y * width + x)) = val4;
}

//blockDim.x must be multiply of 4
__global__ void border_kernel(int *Mandel, int MAX, int grid_off_x, int grid_off_y, int width, int height, float dX, float dY, float x0,
		float y0)
{
	__shared__ int shared_value;

	int frame_size = blockDim.x;
	int off_x = grid_off_x + blockIdx.x * frame_size; //absolute offset of this frame
	int off_y = grid_off_y + blockIdx.y * frame_size;

	int thread_value;
	int pos_x, pos_y; //absolute position of computed pixel on border

	switch(threadIdx.y)
	{
	case 0:
		pos_x = (off_x + threadIdx.x);
		pos_y = off_y;
		break;
	case 1:
		pos_x = (off_x + frame_size - 1);
		pos_y = (off_y + threadIdx.x);
		break;
	case 2:
		pos_x = (off_x + threadIdx.x);
		pos_y = (off_y + frame_size - 1);
		break;
	case 3:
		pos_x = off_x;
		pos_y = (off_y + threadIdx.x);
		break;
	default:
		printf("blockDim.y > 4!\n");
		return;

	}
	thread_value = compute_point(dX * pos_x + x0, dY * pos_y + y0, MAX);

	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0)
		shared_value = thread_value;
	__syncthreads();
	if(shared_value != thread_value && pos_x < width && pos_y < height)
		shared_value = -1;
	__syncthreads();

	if(threadIdx.x == 0 && threadIdx.y == 0)
	{
		if(shared_value != -1)
		{

			int tx = frame_size / 4;
			int ty = MAX_THREADS / tx;
			dim3 threads(tx, ty);

			fill_kernel2<<<1, threads>>>(Mandel, thread_value, off_x, off_y, width, height, frame_size);
			//cudaDeviceSynchronize();

		}

		else if(frame_size <= 32)
		{
			dim3 threads(frame_size, frame_size);
			//printf("m1 %d %d > %d %d\n", threads.x, threads.y, blocks.x, blocks.y);
			mandelbrot1<<<1, threads>>>(Mandel, off_x, off_y, width, height, x0, y0, dX, dY, MAX);
			//cudaDeviceSynchronize();
		}
		else
		{
			dim3 threads(frame_size / 2, 4);
			dim3 blocks(2, 2);
			border_kernel<<<blocks, threads>>>(Mandel, MAX, off_x, off_y, width, height, dX, dY, x0, y0);
			//cudaDeviceSynchronize();

		}

	}

}

void mandelbrot2(int *Mandel, int width, int height, int MAX, float x0, float y0, float x1, float y1)
{
	float dX = (x1 - x0) / (width - 1);
	float dY = (y1 - y0) / (height - 1);

	dim3 blocks((width + INIT_FRAME - 1) / INIT_FRAME, (height + INIT_FRAME - 1) / INIT_FRAME); //cover all picture
	dim3 threads(INIT_FRAME, 4);
	border_kernel<<<blocks, threads>>>(Mandel, MAX, 0, 0, width, height, dX, dY, x0, y0);
	cudaDeviceSynchronize();
	return;
}

#endif /* MANDELBROT_CUH_ */
