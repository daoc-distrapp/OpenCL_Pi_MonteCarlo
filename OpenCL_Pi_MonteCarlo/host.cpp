
#include <CL/cl.h>
#include <vector>
#include <time.h>

int main() {
	// Inicialización general de OpenCL
	printf("Iniciamos Pi Monte Carlo\n");
	// Cuántas plataformas (drivers, implementaciones, ...) hay?
	cl_uint numPlatforms;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	// Obtiene las IDs de todas las plataformas
	std::vector<cl_platform_id> platforms(numPlatforms);
	clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
	// Usamos la plataforma por defecto
	int plat = 0;
	// Crea el contexto
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[plat], 0 };
	cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
	// Obtiene el dispositivo
	cl_device_id device;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
	// Crea la cola de comandos (cola para las kernels)
	const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, device, properties, NULL);

	// Lee el código fuente de la kernel desde el archivo
	char* source = NULL;
	size_t sourceSize = 0;
	FILE* fp = NULL;
	fopen_s(&fp, "PiMonteCarlo.cl", "rb");
	fseek(fp, 0, SEEK_END);
	sourceSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	source = new char[sourceSize];
	fread(source, 1, sourceSize, fp);

	// Crea el programa OpenCL y lo compila, a partir del código fuente de la kernel
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, NULL);
	clBuildProgram(program, 1, &device, "", NULL, NULL);

	// Crea la kernel que se va a ejecutar, a partir del programa ya compilado
	cl_kernel kernel = clCreateKernel(program, "piMC", NULL);

	// Crea los argumentos en el host
	cl_uint seedOffset = (cl_uint)clock();
	cl_uint totalIters = 1024 * 1024 * 1024;

	// Crea el buffer en la GPU, para el valor de salida
	cl_mem gpuBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint), NULL, NULL);

	// Pasa los argumentos a la kernel
	clSetKernelArg(kernel, 0, sizeof(cl_uint), (void *)&seedOffset);
	clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *)&totalIters);
	clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *)&gpuBuffer);

	// Ejecuta la kernel
	size_t localSize = 64; //tamaño del bloque de ejecución
	size_t globalSize = 1024; //número total de threads
	clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	clFinish(commandQueue);

	// Recupera el resultado desde la GPU y lo pone en un buffer del host
	cl_uint* output = (cl_uint*)malloc(sizeof(cl_uint));
	clEnqueueReadBuffer(commandQueue, gpuBuffer, CL_TRUE, 0, sizeof(cl_uint), output, 0, NULL, NULL);

	// Presenta el resultado
	printf("Puntos totales calculados %d\n", totalIters);
	printf("Puntos en el circulo %d\n", output[0]);
	printf("PI aproximado %.20f\n", 4.0 * ((double)output[0] / (double)totalIters));

	// Libera los recursos
	delete[] source;
	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseMemObject(gpuBuffer);
	free(output);

	// Fin !!!
	system("pause");
	return 0;
}