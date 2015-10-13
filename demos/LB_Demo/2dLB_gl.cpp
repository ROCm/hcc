////////////////////////////////////////////////////////////////////////////////
// Crude 2D Lattice Boltzmann Demo program
// CUDA version
// Graham Pullan - Oct 2008
//
// This is a 9 velocity set method:
// Distribution functions are stored as "f" arrays
// Think of these as the number of particles moving in these directions:
//
//      f6  f2   f5
//        \  |  /
//         \ | /
//          \|/
//      f3---|--- f1
//          /|\
//         / | \       and f0 for the rest (zero) velocity
//        /  |  \
//      f7  f4   f8
//
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <hip.h>

#define TILE_I 16
#define TILE_J 8
#define I2D(ni,i,j) (((ni)*(j)) + i)

////////////////////////////////////////////////////////////////////////////////

// OpenGL pixel buffer object and texture //
GLuint gl_PBO, gl_Tex;

// arrays on host //
float *f0, *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8, *plot;
int *solid; 
unsigned int *cmap_rgba, *plot_rgba; // rgba arrays for plotting

// arrays on device //
float *f0_data, *f1_data, *f2_data, *f3_data, *f4_data;
float *f5_data, *f6_data, *f7_data, *f8_data, *plot_data;
int *solid_data;
unsigned int *cmap_rgba_data, *plot_rgba_data;

// textures on device //
// FIXME: templates
// texture<float, 2>
texture f1_tex, f2_tex, f3_tex, f4_tex,
                  f5_tex, f6_tex, f7_tex, f8_tex;

// CUDA special format arrays on device //
hipArray *f1_array, *f2_array, *f3_array, *f4_array; 
hipArray *f5_array, *f6_array, *f7_array, *f8_array; 

// scalars //
float tau,faceq1,faceq2,faceq3;
float vxin, roout;
float width, height;
float minvar, maxvar;

int ni,nj;
int nsolid, nstep, nsteps, ncol;
int ipos_old,jpos_old,draw_solid_flag;

size_t pitch;

// FPS variables
int frameCount = 0;
float fps = 0;
int currentTime = 0;
int previousTime = 0;
void calculateFPS();

////////////////////////////////////////////////////////////////////////////////

//
// OpenGL function prototypes 
//
void display(void);
void resize(int w, int h);
void mouse(int button, int state, int x, int y);
void mouse_motion(int x, int y);

//
// CUDA kernel prototypes
//
__KERNEL void stream_kernel (grid_launch_parm lp, int pitch, float *f1_data, float *f2_data,
                               float *f3_data, float *f4_data, float *f5_data, float *f6_data,
                               float *f7_data, float *f8_data
#ifdef USE_CUDA // CUDA does not support passing textures as paramters, but Kalmar requries it
    );
#else
             ,
             texture f1_tex, texture f2_tex, texture f3_tex, texture f4_tex,
             texture f5_tex, texture f6_tex, texture f7_tex, texture f8_tex);
#endif

__KERNEL void collide_kernel (grid_launch_parm lp, int pitch, float tau, float faceq1, float faceq2, float faceq3,
                                float *f0_data, float *f1_data, float *f2_data,
                                float *f3_data, float *f4_data, float *f5_data, float *f6_data,
                                float *f7_data, float *f8_data, float *plot_data);

__KERNEL void apply_Periodic_BC_kernel (grid_launch_parm lp, int ni, int nj, int pitch, 
					  float *f2_data, float *f4_data, float *f5_data, 
					  float *f6_data, float *f7_data, float *f8_data);

__KERNEL void apply_BCs_kernel (grid_launch_parm lp, int ni, int nj, int pitch, float vxin, float roout,
                                  float faceq2, float faceq3,
                                  float *f0_data, float *f1_data, float *f2_data,
                                  float *f3_data, float *f4_data, float *f5_data, 
                                  float *f6_data, float *f7_data, float *f8_data, int* solid_data);

__KERNEL void get_rgba_kernel (grid_launch_parm lp, int pitch, int ncol, float minvar, float maxvar,
                                 float *plot_data,
                                 unsigned int *plot_rgba_data,
                                 unsigned int *cmap_rgba_data,
                                 int *solid_data);

//
// CUDA kernel C wrappers
//
void stream(void);
void collide(void);
void apply_Periodic_BC(void);
void apply_BCs(void);
void get_rgba(void);  

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    int totpoints,i;
    float rcol,gcol,bcol;

    FILE *fp_col;
    hipChannelFormatDesc desc;

    // The following parameters are usually read from a file, but
    // hard code them for the demo:
    ni=320;
    nj=112;
    vxin=0.04;
    roout=1.0;
    tau=0.51;
    // End of parameter list
    
    // Write parameters to screen
    printf ("ni = %d\n", ni);
    printf ("nj = %d\n", nj);
    printf ("vxin = %f\n", vxin);
    printf ("roout = %f\n", roout);
    printf ("tau = %f\n", tau);
    
    totpoints=ni*nj;

    //
    // allocate memory on host 
    //
    f0 = (float *)malloc(ni*nj*sizeof(float));
    f1 = (float *)malloc(ni*nj*sizeof(float));
    f2 = (float *)malloc(ni*nj*sizeof(float));
    f3 = (float *)malloc(ni*nj*sizeof(float));
    f4 = (float *)malloc(ni*nj*sizeof(float));
    f5 = (float *)malloc(ni*nj*sizeof(float));
    f6 = (float *)malloc(ni*nj*sizeof(float));
    f7 = (float *)malloc(ni*nj*sizeof(float));
    f8 = (float *)malloc(ni*nj*sizeof(float));
    plot = (float *)malloc(ni*nj*sizeof(float));

    solid = (int *)malloc(ni*nj*sizeof(int));
    
    plot_rgba = (unsigned int*)malloc(ni*nj*sizeof(unsigned int));

    //
    // allocate memory on device
    //
    CUDA_SAFE_CALL(hipMallocPitch((void **)&f0_data, &pitch, 
                                   sizeof(float)*ni, nj));
    CUDA_SAFE_CALL(hipMallocPitch((void **)&f1_data, &pitch, 
                                   sizeof(float)*ni, nj));
    CUDA_SAFE_CALL(hipMallocPitch((void **)&f2_data, &pitch, 
                                   sizeof(float)*ni, nj));
    CUDA_SAFE_CALL(hipMallocPitch((void **)&f3_data, &pitch, 
                                   sizeof(float)*ni, nj));
    CUDA_SAFE_CALL(hipMallocPitch((void **)&f4_data, &pitch, 
                                   sizeof(float)*ni, nj));
    CUDA_SAFE_CALL(hipMallocPitch((void **)&f5_data, &pitch, 
                                   sizeof(float)*ni, nj));
    CUDA_SAFE_CALL(hipMallocPitch((void **)&f6_data, &pitch, 
                                   sizeof(float)*ni, nj));
    CUDA_SAFE_CALL(hipMallocPitch((void **)&f7_data, &pitch, 
                                   sizeof(float)*ni, nj));
    CUDA_SAFE_CALL(hipMallocPitch((void **)&f8_data, &pitch, 
                                   sizeof(float)*ni, nj));
    CUDA_SAFE_CALL(hipMallocPitch((void **)&plot_data, &pitch, 
                                   sizeof(float)*ni, nj));


    CUDA_SAFE_CALL(hipMallocPitch((void **)&solid_data, &pitch, 
                                   sizeof(int)*ni, nj));

// FIXME: templatize this
//    desc = hipCreateChannelDesc<float>();
    desc = hipCreateChannelDesc();
    CUDA_SAFE_CALL(hipMallocArray(&f1_array, &desc, ni, nj));
    CUDA_SAFE_CALL(hipMallocArray(&f2_array, &desc, ni, nj));
    CUDA_SAFE_CALL(hipMallocArray(&f3_array, &desc, ni, nj));
    CUDA_SAFE_CALL(hipMallocArray(&f4_array, &desc, ni, nj));
    CUDA_SAFE_CALL(hipMallocArray(&f5_array, &desc, ni, nj));
    CUDA_SAFE_CALL(hipMallocArray(&f6_array, &desc, ni, nj));
    CUDA_SAFE_CALL(hipMallocArray(&f7_array, &desc, ni, nj));
    CUDA_SAFE_CALL(hipMallocArray(&f8_array, &desc, ni, nj));

    //
    // Some factors used in equilibrium f's 
    //
    faceq1 = 4.f/9.f;
    faceq2 = 1.f/9.f;
    faceq3 = 1.f/36.f;

    //
    // Initialise f's
    //
    for (i=0; i<totpoints; i++) {
	f0[i] = faceq1 * roout * (1.f                             - 1.5f*vxin*vxin);
	f1[i] = faceq2 * roout * (1.f + 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
	f2[i] = faceq2 * roout * (1.f                             - 1.5f*vxin*vxin);
	f3[i] = faceq2 * roout * (1.f - 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
	f4[i] = faceq2 * roout * (1.f                             - 1.5f*vxin*vxin);
	f5[i] = faceq3 * roout * (1.f + 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
	f6[i] = faceq3 * roout * (1.f - 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
	f7[i] = faceq3 * roout * (1.f - 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
	f8[i] = faceq3 * roout * (1.f + 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
	plot[i] = vxin;
	solid[i] = 1;
    }

    //
    // Read in colourmap data for OpenGL display 
    //
    fp_col = fopen("cmap.dat","r");
    if (fp_col==NULL) {
	printf("Error: can't open cmap.dat \n");
	return 1;
    }
    
    fscanf (fp_col, "%d", &ncol);
    cmap_rgba = (unsigned int *)malloc(ncol*sizeof(unsigned int));
    CUDA_SAFE_CALL(hipMalloc((void **)&cmap_rgba_data, 
                                   sizeof(unsigned int)*ncol));
    
    for (i=0;i<ncol;i++){
	fscanf(fp_col, "%f%f%f", &rcol, &gcol, &bcol);
	cmap_rgba[i]=((int)(255.0f) << 24) | // convert colourmap to int
	    ((int)(bcol * 255.0f) << 16) |
	    ((int)(gcol * 255.0f) <<  8) |
	    ((int)(rcol * 255.0f) <<  0);
    }
    fclose(fp_col);

    CUDA_SAFE_CALL(hipMalloc((void**)&plot_rgba_data, ni*nj*sizeof(unsigned int)));

    //
    // Transfer initial data to device
    //
    CUDA_SAFE_CALL(hipMemcpy2D((void *)f0_data, pitch, (void *)f0,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)f1_data, pitch, (void *)f1,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)f2_data, pitch, (void *)f2,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)f3_data, pitch, (void *)f3,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)f4_data, pitch, (void *)f4,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)f5_data, pitch, (void *)f5,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)f6_data, pitch, (void *)f6,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)f7_data, pitch, (void *)f7,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)f8_data, pitch, (void *)f8,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)plot_data, pitch, (void *)plot,
                                sizeof(float)*ni,sizeof(float)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy2D((void *)solid_data, pitch, (void *)solid,
                                sizeof(int)*ni,sizeof(int)*ni, nj,
                                hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy((void *)cmap_rgba_data,
                              (void *)cmap_rgba, sizeof(unsigned int)*ncol,
                              hipMemcpyHostToDevice));

    //
    // Iinitialise OpenGL display - use glut
    //
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(ni, nj);                   // Window of ni x nj pixels
    glutInitWindowPosition(50, 50);               // Window position
    glutCreateWindow("GridLaunch 2D LB");               // Window title

    printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
    if(!glewIsSupported(
                        "GL_VERSION_2_0 " 
                        "GL_ARB_pixel_buffer_object "
                        "GL_EXT_framebuffer_object "
                        )){
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return 1;
    }

    // Set up view
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0,ni,0.,nj, -200.0, 200.0);

    // Create texture and bind to gl_Tex
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);                     // Generate 2D texture
    glBindTexture(GL_TEXTURE_2D, gl_Tex);          // bind to gl_Tex
    // texture properties:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ni, nj, 0, 
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
//    printf("Texture created.\n");

    // Create pixel buffer object and bind to gl_PBO
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
//    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, pitch*nj, NULL, GL_STREAM_COPY);
//    CUDA_SAFE_CALL( cudaGLRegisterBufferObject(gl_PBO) );
    printf("Buffer created.\n");
    
    printf("Starting GLUT main loop...\n");
    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutIdleFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(mouse_motion);
     
    glutMainLoop();

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


__KERNEL void stream_kernel (grid_launch_parm lp, int pitch, float *f1_data, float *f2_data,
                               float *f3_data, float *f4_data, float *f5_data,
			       float *f6_data, float *f7_data, float *f8_data
#ifdef USE_CUDA
    )
#else
             ,
             texture f1_tex, texture f2_tex, texture f3_tex, texture f4_tex,
             texture f5_tex, texture f6_tex, texture f7_tex, texture f8_tex)
#endif
// CUDA kernel

{
    GRID_LAUNCH_INIT(lp);

    int i, j, i2d;

    i = lp.groupId.x*TILE_I + lp.threadId.x;
    j = lp.groupId.y*TILE_J + lp.threadId.y;

    i2d = i + j*pitch/sizeof(float);

    // look up the adjacent f's needed for streaming using textures
    // i.e. gather from textures, write to device memory: f1_data, etc
    f1_data[i2d] = tex2D(f1_tex, (float) (i-1)  , (float) j);
    f2_data[i2d] = tex2D(f2_tex, (float) i      , (float) (j-1));
    f3_data[i2d] = tex2D(f3_tex, (float) (i+1)  , (float) j);
    f4_data[i2d] = tex2D(f4_tex, (float) i      , (float) (j+1));
    f5_data[i2d] = tex2D(f5_tex, (float) (i-1)  , (float) (j-1));
    f6_data[i2d] = tex2D(f6_tex, (float) (i+1)  , (float) (j-1));
    f7_data[i2d] = tex2D(f7_tex, (float) (i+1)  , (float) (j+1));
    f8_data[i2d] = tex2D(f8_tex, (float) (i-1)  , (float) (j+1));
}

void stream(void)

// C wrapper

{
    // Device-to-device mem-copies to transfer data from linear memory (f1_data)
    // to CUDA format memory (f1_array) so we can use these in textures
    CUDA_SAFE_CALL(hipMemcpy2DToArray(f1_array, 0, 0, (void *)f1_data, pitch,
                                       sizeof(float)*ni, nj,
                                       hipMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(hipMemcpy2DToArray(f2_array, 0, 0, (void *)f2_data, pitch,
                                       sizeof(float)*ni, nj,
                                       hipMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(hipMemcpy2DToArray(f3_array, 0, 0, (void *)f3_data, pitch,
                                       sizeof(float)*ni, nj,
                                       hipMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(hipMemcpy2DToArray(f4_array, 0, 0, (void *)f4_data, pitch,
                                       sizeof(float)*ni, nj,
                                       hipMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(hipMemcpy2DToArray(f5_array, 0, 0, (void *)f5_data, pitch,
                                       sizeof(float)*ni, nj,
                                       hipMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(hipMemcpy2DToArray(f6_array, 0, 0, (void *)f6_data, pitch,
                                       sizeof(float)*ni, nj,
                                       hipMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(hipMemcpy2DToArray(f7_array, 0, 0, (void *)f7_data, pitch,
                                       sizeof(float)*ni, nj,
                                       hipMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(hipMemcpy2DToArray(f8_array, 0, 0, (void *)f8_data, pitch,
                                       sizeof(float)*ni, nj,
                                       hipMemcpyDeviceToDevice));


    // Tell CUDA that we want to use f1_array etc as textures. Also
    // define what type of interpolation we want (nearest point)
    f1_tex.filterMode = hipFilterModePoint;
    CUDA_SAFE_CALL(hipBindTextureToArray(f1_tex, f1_array));

    f2_tex.filterMode = hipFilterModePoint;
    CUDA_SAFE_CALL(hipBindTextureToArray(f2_tex, f2_array));

    f3_tex.filterMode = hipFilterModePoint;
    CUDA_SAFE_CALL(hipBindTextureToArray(f3_tex, f3_array));

    f4_tex.filterMode = hipFilterModePoint;
    CUDA_SAFE_CALL(hipBindTextureToArray(f4_tex, f4_array));

    f5_tex.filterMode = hipFilterModePoint;
    CUDA_SAFE_CALL(hipBindTextureToArray(f5_tex, f5_array));

    f6_tex.filterMode = hipFilterModePoint;
    CUDA_SAFE_CALL(hipBindTextureToArray(f6_tex, f6_array));

    f7_tex.filterMode = hipFilterModePoint;
    CUDA_SAFE_CALL(hipBindTextureToArray(f7_tex, f7_array));

    f8_tex.filterMode = hipFilterModePoint;
    CUDA_SAFE_CALL(hipBindTextureToArray(f8_tex, f8_array));

    dim3 grid = DIM3(ni/TILE_I, nj/TILE_J);
    dim3 block = DIM3(TILE_I, TILE_J);

    hipLaunchKernel(stream_kernel, grid, block, pitch, f1_data, f2_data, f3_data, f4_data,
                                   f5_data, f6_data, f7_data, f8_data
#ifdef USE_CUDA
        );
#else
             ,
             f1_tex, f2_tex, f3_tex, f4_tex,
             f5_tex, f6_tex, f7_tex, f8_tex);
#endif

    CUT_CHECK_ERROR("stream failed.");

    CUDA_SAFE_CALL(hipUnbindTexture(f1_tex));
    CUDA_SAFE_CALL(hipUnbindTexture(f2_tex));
    CUDA_SAFE_CALL(hipUnbindTexture(f3_tex));
    CUDA_SAFE_CALL(hipUnbindTexture(f4_tex));
    CUDA_SAFE_CALL(hipUnbindTexture(f5_tex));
    CUDA_SAFE_CALL(hipUnbindTexture(f6_tex));
    CUDA_SAFE_CALL(hipUnbindTexture(f7_tex));
    CUDA_SAFE_CALL(hipUnbindTexture(f8_tex));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__KERNEL void collide_kernel (grid_launch_parm lp, int pitch, float tau, float faceq1, float faceq2, float faceq3,
                                float *f0_data, float *f1_data, float *f2_data,
                                float *f3_data, float *f4_data, float *f5_data, float *f6_data,
                                float *f7_data, float *f8_data, float *plot_data)
// CUDA kernel

{
    GRID_LAUNCH_INIT(lp);

    int i, j, i2d;
    float ro, vx, vy, v_sq_term, rtau, rtau1;
    float f0now, f1now, f2now, f3now, f4now, f5now, f6now, f7now, f8now;
    float f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq;
    
    
    i = lp.groupId.x*TILE_I + lp.threadId.x;
    j = lp.groupId.y*TILE_J + lp.threadId.y;

    i2d = i + j*pitch/sizeof(float);

    rtau = 1.f/tau;
    rtau1 = 1.f - rtau;    

    // Read all f's and store in registers
    f0now = f0_data[i2d];
    f1now = f1_data[i2d];
    f2now = f2_data[i2d];
    f3now = f3_data[i2d];
    f4now = f4_data[i2d];
    f5now = f5_data[i2d];
    f6now = f6_data[i2d];
    f7now = f7_data[i2d];
    f8now = f8_data[i2d];

    // Macroscopic flow props:
    ro =  f0now + f1now + f2now + f3now + f4now + f5now + f6now + f7now + f8now;
    vx = (f1now - f3now + f5now - f6now - f7now + f8now)/ro;
    vy = (f2now - f4now + f5now + f6now - f7now - f8now)/ro;

    // Set plotting variable to velocity magnitude
    plot_data[i2d] = SQRTF(vx*vx + vy*vy);
    
    // Calculate equilibrium f's
    v_sq_term = 1.5f*(vx*vx + vy*vy);
    f0eq = ro * faceq1 * (1.f - v_sq_term);
    f1eq = ro * faceq2 * (1.f + 3.f*vx + 4.5f*vx*vx - v_sq_term);
    f2eq = ro * faceq2 * (1.f + 3.f*vy + 4.5f*vy*vy - v_sq_term);
    f3eq = ro * faceq2 * (1.f - 3.f*vx + 4.5f*vx*vx - v_sq_term);
    f4eq = ro * faceq2 * (1.f - 3.f*vy + 4.5f*vy*vy - v_sq_term);
    f5eq = ro * faceq3 * (1.f + 3.f*(vx + vy) + 4.5f*(vx + vy)*(vx + vy) - v_sq_term);
    f6eq = ro * faceq3 * (1.f + 3.f*(-vx + vy) + 4.5f*(-vx + vy)*(-vx + vy) - v_sq_term);
    f7eq = ro * faceq3 * (1.f + 3.f*(-vx - vy) + 4.5f*(-vx - vy)*(-vx - vy) - v_sq_term);
    f8eq = ro * faceq3 * (1.f + 3.f*(vx - vy) + 4.5f*(vx - vy)*(vx - vy) - v_sq_term);

    // Do collisions
    f0_data[i2d] = rtau1 * f0now + rtau * f0eq;
    f1_data[i2d] = rtau1 * f1now + rtau * f1eq;
    f2_data[i2d] = rtau1 * f2now + rtau * f2eq;
    f3_data[i2d] = rtau1 * f3now + rtau * f3eq;
    f4_data[i2d] = rtau1 * f4now + rtau * f4eq;
    f5_data[i2d] = rtau1 * f5now + rtau * f5eq;
    f6_data[i2d] = rtau1 * f6now + rtau * f6eq;
    f7_data[i2d] = rtau1 * f7now + rtau * f7eq;
    f8_data[i2d] = rtau1 * f8now + rtau * f8eq;
}

void collide(void)

// C wrapper

{
    dim3 grid = DIM3(ni/TILE_I, nj/TILE_J);
    dim3 block = DIM3(TILE_I, TILE_J);

    hipLaunchKernel(collide_kernel, grid, block, pitch, tau, faceq1, faceq2, faceq3,
                                    f0_data, f1_data, f2_data, f3_data, f4_data,
                                    f5_data, f6_data, f7_data, f8_data, plot_data);
    
    CUT_CHECK_ERROR("collide failed.");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__KERNEL void apply_BCs_kernel (grid_launch_parm lp, int ni, int nj, int pitch, float vxin, float roout,
                                  float faceq2, float faceq3,
                                  float *f0_data, float *f1_data, float *f2_data,
                                  float *f3_data, float *f4_data, float *f5_data, 
                                  float *f6_data, float *f7_data, float *f8_data,
				  int* solid_data)

// CUDA kernel all BC's apart from periodic boundaries:

{
    GRID_LAUNCH_INIT(lp);

    int i, j, i2d, i2d2;
    float v_sq_term;
    float f1old, f2old, f3old, f4old, f5old, f6old, f7old, f8old;
    
    i = lp.groupId.x*TILE_I + lp.threadId.x;
    j = lp.groupId.y*TILE_J + lp.threadId.y;

    i2d = i + j*pitch/sizeof(float);

    // Solid BC: "bounce-back"
    if (solid_data[i2d] == 0) {
      f1old = f1_data[i2d];
      f2old = f2_data[i2d];
      f3old = f3_data[i2d];
      f4old = f4_data[i2d];
      f5old = f5_data[i2d];
      f6old = f6_data[i2d];
      f7old = f7_data[i2d];
      f8old = f8_data[i2d];
      
      f1_data[i2d] = f3old;
      f2_data[i2d] = f4old;
      f3_data[i2d] = f1old;
      f4_data[i2d] = f2old;
      f5_data[i2d] = f7old;
      f6_data[i2d] = f8old;
      f7_data[i2d] = f5old;
      f8_data[i2d] = f6old;
    }


    // Inlet BC - very crude
    if (i == 0) {
      v_sq_term = 1.5f*(vxin * vxin);
      
      f1_data[i2d] = roout * faceq2 * (1.f + 3.f*vxin + 3.f*v_sq_term);
      f5_data[i2d] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);
      f8_data[i2d] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);

    }
        
    // Exit BC - very crude
    if (i == (ni-1)) {
      i2d2 = i2d - 1;
      f3_data[i2d] = f3_data[i2d2];
      f6_data[i2d] = f6_data[i2d2];
      f7_data[i2d] = f7_data[i2d2];

    }
}

void apply_BCs(void)

// C wrapper

{
    dim3 grid = DIM3(ni/TILE_I, nj/TILE_J);
    dim3 block = DIM3(TILE_I, TILE_J);

    hipLaunchKernel(apply_BCs_kernel, grid, block, ni, nj, pitch, vxin, roout, faceq2,faceq3,
                                      f0_data, f1_data, f2_data,
                                      f3_data, f4_data, f5_data, 
                                      f6_data, f7_data, f8_data, solid_data);
    
    CUT_CHECK_ERROR("apply_BCs failed.");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__KERNEL void apply_Periodic_BC_kernel (grid_launch_parm lp, int ni, int nj, int pitch, 
					  float *f2_data, float *f4_data, float *f5_data, 
					  float *f6_data, float *f7_data, float *f8_data)
// CUDA kernel

{
    GRID_LAUNCH_INIT(lp);

    int i, j, i2d, i2d2;
    
    i = lp.groupId.x*TILE_I + lp.threadId.x;
    j = lp.groupId.y*TILE_J + lp.threadId.y;

    i2d = i + j*pitch/sizeof(float);

    if (j == 0 ) {
        i2d2 = i + (nj-1)*pitch/sizeof(float);
        f2_data[i2d] = f2_data[i2d2];
        f5_data[i2d] = f5_data[i2d2];
        f6_data[i2d] = f6_data[i2d2];
    }
    if (j == (nj-1)) {
        i2d2 = i;
        f4_data[i2d] = f4_data[i2d2];
        f7_data[i2d] = f7_data[i2d2];
        f8_data[i2d] = f8_data[i2d2];
    }
}

// C wrapper

void apply_Periodic_BC(void)
{
    dim3 grid = DIM3(ni/TILE_I, nj/TILE_J);
    dim3 block = DIM3(TILE_I, TILE_J);

    hipLaunchKernel(apply_Periodic_BC_kernel, grid, block, ni, nj, pitch,
					      f2_data,f4_data, f5_data, 
					      f6_data, f7_data, f8_data);
    
    CUT_CHECK_ERROR("apply_Periodic_BC failed.");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__KERNEL void get_rgba_kernel (grid_launch_parm lp, int pitch, int ncol, float minvar, float maxvar,
                                 float *plot_data,
                                 unsigned int *plot_rgba_data,
                                 unsigned int *cmap_rgba_data,
                                 int *solid_data)

// CUDA kernel to fill plot_rgba_data array for plotting

{
    GRID_LAUNCH_INIT(lp);

    int i, j, i2d, icol;
    float frac;
    
    i = lp.groupId.x*TILE_I + lp.threadId.x;
    j = lp.groupId.y*TILE_J + lp.threadId.y;

    i2d = i + j*pitch/sizeof(float);

    frac = (plot_data[i2d]-minvar)/(maxvar-minvar);
    icol = (int)(frac * (float)ncol);
    plot_rgba_data[i2d] = solid_data[i2d] * cmap_rgba_data[icol];
}

void get_rgba(void)

// C wrapper

{
    dim3 grid = DIM3(ni/TILE_I, nj/TILE_J);
    dim3 block = DIM3(TILE_I, TILE_J);

    hipLaunchKernel(get_rgba_kernel, grid, block, pitch, ncol, minvar, maxvar,
				     plot_data, plot_rgba_data, cmap_rgba_data,
                                     solid_data);
    
    CUT_CHECK_ERROR("get_rgba failed.");
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void display(void)

// This function is called automatically, over and over again,  by GLUT 

{
    // Set upper and lower limits for plotting
    minvar=0.;
    maxvar=0.2;

    // Do one Lattice Boltzmann step: stream, BC, collide:
    stream();
    apply_Periodic_BC();
    apply_BCs();
    collide();

    // For plotting, map the plot_rgba_data array to the
    // gl_PBO pixel buffer
//    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&plot_rgba_data, gl_PBO));

    // Fill the plot_rgba_data array (and the pixel buffer)
    get_rgba();

    // Fill the pixel buffer with the plot_rgba array
    CUDA_SAFE_CALL(hipMemcpy(plot_rgba, plot_rgba_data, ni*nj*sizeof(unsigned int), hipMemcpyDeviceToHost));
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,ni*nj*sizeof(unsigned int),
		 (void **)plot_rgba,GL_STREAM_COPY);

//    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(gl_PBO));

    // Copy the pixel buffer to the texture, ready to display
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,ni,nj,GL_RGBA,GL_UNSIGNED_BYTE,0);

    // Render one quad to the screen and colour it using our texture
    // i.e. plot our plotvar data to the screen
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    glTexCoord2f (0.0, 0.0);
    glVertex3f (0.0, 0.0, 0.0);
    glTexCoord2f (1.0, 0.0);
    glVertex3f (ni, 0.0, 0.0);
    glTexCoord2f (1.0, 1.0);
    glVertex3f (ni, nj, 0.0);
    glTexCoord2f (0.0, 1.0);
    glVertex3f (0.0, nj, 0.0);
    glEnd();
    calculateFPS();
    printf("FPS: %4.2f\r", fps);
    glutSwapBuffers();

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void resize(int w, int h)

// GLUT resize callback to allow us to change the window size

{
   width = w;
   height = h;
   glViewport (0, 0, w, h); 
   glMatrixMode (GL_PROJECTION); 
   glLoadIdentity (); 
   glOrtho (0., ni, 0., nj, -200. ,200.); 
   glMatrixMode (GL_MODELVIEW); 
   glLoadIdentity ();
}
    
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void mouse(int button, int state, int x, int y)

// GLUT mouse callback. Left button draws the solid, right button removes solid

{
    float xx,yy;

    if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
        draw_solid_flag = 0;
        xx=x;
        yy=y;
        ipos_old=xx/width*ni;
        jpos_old=(height-yy)/height*nj;
    }

    if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN)) {
        draw_solid_flag = 1;
        xx=x;
        yy=y;
        ipos_old=xx/width*ni;
        jpos_old=(height-yy)/height*nj;
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void mouse_motion(int x, int y)

// GLUT call back for when the mouse is moving
// This sets the solid array to draw_solid_flag as set in the mouse callback
// It will draw a staircase line if we move more than one pixel since the
// last callback - that makes the coding a bit cumbersome:

{
    float xx,yy,frac;
    int ipos,jpos,i,j,i1,i2,j1,j2, jlast, jnext;
    xx=x;
    yy=y;
    ipos=(int)(xx/width*(float)ni);
    jpos=(int)((height-yy)/height*(float)nj);

    if (ipos <= ipos_old){
        i1 = ipos;
        i2 = ipos_old;
        j1 = jpos;
        j2 = jpos_old;
    }
    else {
        i1 = ipos_old;
        i2 = ipos;
        j1 = jpos_old;
        j2 = jpos;
    }
    
    jlast=j1;

    for (i=i1;i<=i2;i++){
        if (i1 != i2) {
            frac=(float)(i-i1)/(float)(i2-i1);
            jnext=(int)(frac*(j2-j1))+j1;
        }
        else {
            jnext=j2;
        }
        if (jnext >= jlast) {
            solid[I2D(ni,i,jlast)]=draw_solid_flag;
            for (j=jlast; j<=jnext; j++){
                solid[I2D(ni,i,j)]=draw_solid_flag;
            }
        }
        else {
            solid[I2D(ni,i,jlast)]=draw_solid_flag;
            for (j=jnext; j<=jlast; j++){
                solid[I2D(ni,i,j)]=draw_solid_flag;
            }
        }
        jlast = jnext;
    }


    // Copy the solid array (host) to the solid_data array (device)
    CUDA_SAFE_CALL(hipMemcpy2D((void *)solid_data, pitch, (void *)solid,
                                sizeof(int)*ni,sizeof(int)*ni, nj,
                                hipMemcpyHostToDevice));    
    
    ipos_old=ipos;
    jpos_old=jpos;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void calculateFPS()
{
  frameCount++;
  currentTime = glutGet(GLUT_ELAPSED_TIME);

  int timeInterval = currentTime - previousTime;

  if(timeInterval > 1000)
  {
    fps = frameCount / (timeInterval / 1000.0f);
    previousTime = currentTime;
    frameCount = 0;
  }
}

