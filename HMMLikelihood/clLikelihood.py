import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sps
from scipy.sparse import vstack
from scipy.sparse.linalg import spsolve
import pyopencl as cl
import pyopencl.cltypes
import numpy as np
import time
import pandas as pd
import pytwalk
import matplotlib.pylab as plt
import os

def definesFromDict(defines_dic):
    defines = ""
    for item in defines_dic.items():
        defines += "-D "
        exec(str(item[0]) + "=" + str(item[1]))     
        defines += str(item[0]) + "=" + str(locals()[str(item[0])]) + " "
    return defines

def definesToLocals(defines_dic):
    for item in defines_dic.items():
        if isinstance(item[1],str): 
            globals()[item[0]]= int(eval(item[1]))
        else:
            globals()[item[0]] = item[1]

def check_error(l1, l2):
    """
        Error check function for output of OpenCL
    """
    if (l2 == np.nan):
        raise ValueError('NAN')
    if (l2 == np.inf):
        raise ValueError('INF')
    if (np.abs((l2 - l1) / max(l1, l2)) > 1e-5):
        raise ValueError('PRECISION')

class Likelihood(object):
    """ 
    Class to calculate a likelihood function
    Usage
        model = Likelhood(data, transistion_matrix, kernels)
        for [some likelihood criteria]
            model.set_transistion_matrix(transistion_matrix)
            model.set_kernels(kernels)
            likelihood = model()
        
    """
    def __init__(self, data=None, transistion_matrix=None, kernels=None, 
                 mode='cpu', 
                 kernel_type='gaussian', 
                 calc_function=None,
                 file_diag_kernel=None,
                 file_diag_normal=None,
                 file_diag_mat_mul=None,
                 file_matrixmul=None,
                 computeUnits=None,
                 wrkUnit=128,
                 platform_id=0
                ):
        """
        Initialise likehood class
        Data -> 
                    [[data1_param1, data1_param2, ...],
                     [data2_param1, data2_param2, ...]
                     ...
                     [dataT_param1, data1_param2, ...]]
                     
                     Estimate T from data
                     
        transistion_matrix -> 
                    [[a11, a12, ..., a1k],
                     [a11, a12, ..., a1k],
                     ...
                     [ak1, ak2, ..., akk]]
        
        kernels -> 
                    [[kernel1_param1, kernel1_param2, ...],
                     [kernel2_param1, kernel2_param2, ...]
                     ...
                     [kernelK_param1, kernelK_param2, ...]]
                     
                     Estimate number of kernels from kernel parameters.
                     
        mode -> select which function to execute
                set None to use own defined function
                
        kernel_type -> Function to be used for kernel
        
        calc_function -> own defined likelihood function
        
        TODO:
         Error checking
        """
        if (data is None) or (transistion_matrix is None) or (kernels is None):
            print("WARNING: Empty Likelihood function, please re-initialise")
            return

        if (file_diag_kernel==None):
            abs_path = os.path.dirname(os.path.realpath(__file__))
            self.file_diag_kernel = abs_path + "/prob_function.cl"
        else:
            self.file_diag_kernel = file_diag_kernel

        if (file_diag_normal==None):
            abs_path = os.path.dirname(os.path.realpath(__file__))
            self.file_diag_normal = abs_path + "/diag_normalc.cl"
        else:
            self.file_diag_normal = file_diag_normal
            
        if (file_diag_mat_mul==None):
            abs_path = os.path.dirname(os.path.realpath(__file__))
            self.file_diag_mat_mul = abs_path + "/diag_mat_mulO.cl"
        else:
            self.file_diag_mat_mul = file_diag_mat_mul
            
        if (file_matrixmul==None):
            abs_path = os.path.dirname(os.path.realpath(__file__))
            self.file_matrixmul = abs_path + "/matrixmulT.cl"
        else:
            self.file_matrixmul = file_matrixmul

        self.data = np.ascontiguousarray(data, dtype=np.float64)
        
        self.T = data.shape[0]
        self.n_data_dim = data.shape[1]
        
        self.transistion_matrix = np.ascontiguousarray(transistion_matrix, dtype=np.float64)
        
        self.dim = transistion_matrix.shape[0]
        
        self.stationary_matrix = self.stationary_matrix_calculation(self.transistion_matrix)
        #print(self.stationary_matrix)
        self.kernels = np.ascontiguousarray(kernels, dtype=np.float64)
        
        self.n_kernel_param = self.kernels.shape[1]
        
        self.mode = mode
        self.calc_function = calc_function
            
        self.kernel_type = kernel_type
        
        self.pi = None
        
        self.platform_id = platform_id
        if (computeUnits==None):
            self.pyOpenCLInfo(False)
            self.computeUnits = self.recommend_CU
        else:
            self.computeUnits = computeUnits

        self.wrkUnit = wrkUnit
        
        


    def pyOpenCLInfo(self, output_info=True):
        if (output_info):
            print('PyOpenCL version: ' + cl.VERSION_TEXT)
            print('OpenCL header version: ' + '.'.join(map(str, cl.get_cl_header_version())) + '\n')

            # Get installed platforms (SDKs)
            print('- Installed platforms (SDKs) and available devices:')
            platforms = cl.get_platforms()

            for plat in platforms:
                indent = ''

                # Get and print platform info
                print(indent + '{} ({})'.format(plat.name, plat.vendor))
                indent = '\t'
                print(indent + 'Version: ' + plat.version)
                print(indent + 'Profile: ' + plat.profile)
                print(indent + 'Extensions: ' + str(plat.extensions.strip().split(' ')))

                # Get and print device info
                devices = plat.get_devices(cl.device_type.ALL)

                print(indent + 'Available devices: ')
                if not devices:
                    print(indent + '\tNone')

                for dev in devices:
                    indent = '\t\t'
                    print(indent + '{} ({})'.format(dev.name, dev.vendor))

                    indent = '\t\t\t'
                    flags = [('Version', dev.version),
                            ('Type', cl.device_type.to_string(dev.type)),
                            ('Extensions', str(dev.extensions.strip().split(' '))),
                            ('Memory (global)', str(dev.global_mem_size)),
                            ('Memory (local)', str(dev.local_mem_size)),
                            ('Address bits', str(dev.address_bits)),
                            ('Max work item dims', str(dev.max_work_item_dimensions)),
                            ('Max work group size', str(dev.max_work_group_size)),
                            ('Max compute units', str(dev.max_compute_units)),
                            ('Driver version', dev.driver_version),
                            ('Image support', str(bool(dev.image_support))),
                            ('Little endian', str(bool(dev.endian_little))),
                            ('Device available', str(bool(dev.available))),
                            ('Compiler available', str(bool(dev.compiler_available)))]

                    [print(indent + '{0:<25}{1:<10}'.format(name + ':', flag)) for name, flag in flags]

                    # Device version string has the following syntax, extract the number like this
                    # OpenCL<space><major_version.minor_version><space><vendor-specific information>
                    version_number = float(dev.version.split(' ')[1])

                print('')
        else:
            platform = cl.get_platforms()[self.platform_id]
            device = platform.get_devices(cl.device_type.ALL)[0]           
            self.recommend_CU = device.max_compute_units

    def __call__(self, warning=True):
        """ 
        Calculate the Likelihood and return the value
        """
        if self.mode == 'gpu':
            return self.gpu_calculation
        elif self.mode == 'cpu':
            return self.cpu_calculation
        else:
            if (warning):
                print("WARNING: Assigning custom function")
            return self.calc_function        
        
    def set_transistion_matrix(self, transistion_matrix):
        """ 
        Update the transistion matrix 
        """
        self.transistion_matrix = np.ascontiguousarray(transistion_matrix, dtype=np.float64)

    def get_transistion_matrix(self):
        return self.transistion_matrix      
        
    def set_kernels(self, kernels):
        """ 
        Update the transistion matrix 
        """
        self.kernels = np.ascontiguousarray(kernels, dtype=np.float64)    
        
    def get_kernels(self):
        return self.kernels      
        
    def stationary_matrix_calculation(self, P):
        size    = P.shape[0]
        dP      = P - np.eye(size)
        #Replace the first equation by the normalizing condition.
        A       = vstack([np.ones(size), dP.T[1:,:]]).tocsr()
        rhs     = np.zeros((size,))
        rhs[0]  = 1
        pi = spsolve(A, rhs)
        self.pi = pi
        return pi

    def gpu_chain(self, silent=False):
        
        # Calculate binomarl distribution
        self.defines_dic0 = {
        "T"               :str(self.T),
        "dim"             :str(self.dim),
        "computeUnits"    :str(self.computeUnits),
        "wrkUnit"         :str(self.wrkUnit),
        "n_elem"          :"np.ceil((dim*dim+wrkUnit)/wrkUnit).astype(np.int)",   # For temporary result matrix
        "n_wrkGroups"     :"computeUnits",
        "matSize"         :"dim*dim",
        "n_kernel_param"  : str(self.n_kernel_param),
        "n_data_dim"      : str(self.n_data_dim),
        "minVal"          : str(0.000001)
        }
        definesToLocals(self.defines_dic0)
        self.defines0 = definesFromDict(self.defines_dic0) + " -cl-std=CL1.2"

        diag0 = np.zeros((dim, n_kernel_param)).astype(cl.cltypes.float)
        diag0 = self.kernels.astype(cl.cltypes.float)
        #diag0[:,4] = 0
        #data points
        mat0 = np.zeros((T, n_data_dim)).astype(cl.cltypes.float)
        #mat0 = self.data.astype(cl.cltypes.float)
        
        self.res0 = np.zeros((T, dim),np.float32)

        self.code_kernel = open(self.file_diag_kernel, "r").read()
        self.code_diag = open(self.file_diag_normal, "r").read()
        self.code0 = self.code_kernel + "\n" + self.code_diag

        self.platform = cl.get_platforms()[self.platform_id]
        self.device = self.platform.get_devices()[0]
        if (not silent):
            print("Using Device : ", self.device.name)
            print("Compute Units: ", self.computeUnits)
        self.context = cl.Context([self.device])
        self.program0 = cl.Program(self.context, self.code0).build(self.defines0)
        self.queue = cl.CommandQueue(self.context)

        # Buffer creation
        mem_flags = cl.mem_flags
        self.mat_buf0 = cl.Buffer(self.context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=mat0)
        self.diag_buf0 = cl.Buffer(self.context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=diag0)
        self.res_buf0 = cl.Buffer(self.context, mem_flags.READ_WRITE, self.res0.nbytes)

        self.kernel0 = self.program0.diag_normal

        # Set program arguments
        self.globalItems0 = ( T, )
        self.localItems0 = None # (32, )

        self.kernel0.set_arg(0, self.diag_buf0)
        self.kernel0.set_arg(1, self.mat_buf0 )
        self.kernel0.set_arg(2, self.res_buf0 )      
        
        # Diagonal Matrix multiplying

        defines_dic1 = {
        "T"               :str(self.T),
        "dim"             :str(self.dim),
        "computeUnits"    :str(self.computeUnits),
        "wrkUnit"         :str(self.wrkUnit),
        "n_elem"          :"np.ceil((dim*dim+wrkUnit)/wrkUnit).astype(np.int)",   # For temporary result matrix
        "n_wrkGroups"     :"computeUnits",
        "matSize"         :"dim*dim",
        "n_kernel_param"  : str(self.n_kernel_param),
        "n_data_dim"      : str(self.n_data_dim),
        "minVal"          : str(0.000001)
        }
        definesToLocals(defines_dic1)
        self.defines1 = definesFromDict(defines_dic1)+ " -cl-std=CL1.2"

        # For use with diag_mat_mulB
        #n_dat_mat1 = np.zeros(wrkUnit*computeUnits + 1).astype(np.int)
        #n_dat_mat1[1:] = int(T / (wrkUnit*computeUnits))
        #n_dat_mat1[1:T%(wrkUnit*computeUnits) + 1] += 1
        #n_dat_mat1 = np.cumsum(n_dat_mat1)
        #n_dat_mat1 = n_dat_mat1.astype(cl.cltypes.int)
        #self.n_dat_mat1 = n_dat_mat1

        n_dat_mat1 = np.zeros(n_wrkGroups + 1).astype(np.int)
        n_dat_mat1[1:] = int((T) / n_wrkGroups)
        n_dat_mat1[1:(T )%n_wrkGroups + 1] += 1
        n_dat_mat1 = np.cumsum(n_dat_mat1)
        n_dat_mat1 = n_dat_mat1.astype(cl.cltypes.int)
        self.n_dat_mat1 = n_dat_mat1        
        
        # transistion matrix
        mat1 = np.zeros((dim, dim)).astype(cl.cltypes.float)
        mat1 = self.transistion_matrix.astype(cl.cltypes.float)
        # Calculated from data and kernels - taken from previous kernel
        #diag1 = np.random.random((T, dim)).astype(cl.cltypes.float)
        # Result
        self.res1 = np.zeros((T,dim,dim),np.float32)

        self.code1 = open(self.file_diag_mat_mul, "r").read()

        self.program1 = cl.Program(self.context, self.code1).build(self.defines1)

        # Buffer creation
        mem_flags = cl.mem_flags
        self.n_dat_mat_buf1 = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=n_dat_mat1)
        self.mat_buf1 = cl.Buffer(self.context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=mat1)
        #self.diag_buf1 = cl.Buffer(self.context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=diag1)
        self.res_buf1 = cl.Buffer(self.context, mem_flags.READ_WRITE, self.res1.nbytes)

        self.kernel1 = self.program1.diag_mat_mul

        # Set program arguments
        # For use with diag_mat_mulB
        #self.globalItems1 = ( computeUnits*wrkUnit, )
        #self.localItems1 = None # (32, )
        self.globalItems1 = ( computeUnits*dim, )
        self.localItems1 = ( dim, )
        #print(self.globalItems1)

        self.kernel1.set_arg(0, self.n_dat_mat_buf1)
        self.kernel1.set_arg(1, self.mat_buf1 )   # Transistion matrix
        self.kernel1.set_arg(2, self.res_buf0 )   # Diagonal matrix values from previous step
        self.kernel1.set_arg(3, self.res_buf1 )

        # MatrixMatrix Multiplying
        
        self.defines_dic2 = {
        "T"               :str(self.T),
        "dim"             :str(self.dim),
        "computeUnits"    :str(self.computeUnits),
        "wrkUnit"         :str(self.wrkUnit),
        "n_elem"          :"np.ceil((dim*dim+wrkUnit)/wrkUnit).astype(np.int)",   # For temporary result matrix
        "n_wrkGroups"     :"computeUnits",
        "matSize"         :"dim*dim",
        "n_kernel_param"  : str(self.n_kernel_param),
        "n_data_dim"      : str(self.n_data_dim),
        "minVal"          : str(0.000001)
        }
        definesToLocals(self.defines_dic2)
        self.defines2 = definesFromDict(self.defines_dic2) + " -cl-std=CL1.2"

        n_element_mat2 = np.zeros(wrkUnit + 1).astype(np.int)
        n_element_mat2[1:] = int(dim*dim / wrkUnit)
        n_element_mat2[1:(dim*dim + 1)%wrkUnit] += 1
        n_element_mat2 = np.cumsum(n_element_mat2)
        n_element_mat2 = n_element_mat2.astype(cl.cltypes.int)
        self.n_element_mat2 = n_element_mat2

        n_mat_mat2 = np.zeros(n_wrkGroups + 1).astype(np.int)
        n_mat_mat2[1:] = int((T) / n_wrkGroups)
        n_mat_mat2[1:(T )%n_wrkGroups + 1] += 1
        n_mat_mat2 = np.cumsum(n_mat_mat2)
        n_mat_mat2 = n_mat_mat2.astype(cl.cltypes.int)
        self.n_mat_mat2 = n_mat_mat2

        # Matrix input - already made as output from previous step
        #mat2 = np.random.random((T, dim, dim)).astype(cl.cltypes.float) / T
        # Matrix outputs 
        self.res2 = np.zeros((computeUnits,dim, dim),np.float32)
        # Scaling coefficients used in matrix multiplication
        self.resCoef2 = np.zeros((computeUnits),np.int32)

        self.code2 = open(self.file_matrixmul, "r").read()
        self.program2 = cl.Program(self.context, self.code2).build(self.defines2)

        # Buffer creation
        self.n_element_mat_buf2 = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.n_element_mat2)
        self.n_mat_mat_buf2     = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=self.n_mat_mat2)
        #mat_buf2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=mat2)
        self.res_buf2           = cl.Buffer(self.context, mem_flags.READ_WRITE, self.res2.nbytes)
        self.resCoef_buf2       = cl.Buffer(self.context, mem_flags.READ_WRITE, self.resCoef2.nbytes)

        self.kernel2 = self.program2.matrixmul

        # Set program arguments
        self.globalItems2 = ( computeUnits*wrkUnit, )
        self.localItems2  = ( wrkUnit, )

        self.kernel2.set_arg(0, self.n_element_mat_buf2)
        self.kernel2.set_arg(1, self.n_mat_mat_buf2)
        #self.kernel2.set_arg(2, self.mat_buf2 )
        self.kernel2.set_arg(2, self.res_buf1 ) # input argument for MM
        self.kernel2.set_arg(3, self.res_buf2 )
        self.kernel2.set_arg(4, self.resCoef_buf2 )

        cl.enqueue_copy(self.queue, self.mat_buf0, self.data.astype(cl.cltypes.float), is_blocking=True)
        cl.enqueue_copy(self.queue, self.mat_buf1, self.transistion_matrix.astype(cl.cltypes.float), is_blocking=True) 
        cl.enqueue_copy(self.queue, self.diag_buf0, self.kernels.astype(cl.cltypes.float), is_blocking=True) 
     
    def gpu_update_kernels(self, kernels):
        self.kernels = np.ascontiguousarray(kernels, dtype=np.float64)
        cl.enqueue_copy(self.queue, self.diag_buf0, self.kernels.astype(cl.cltypes.float), is_blocking=True) 
        
    def gpu_update_transistion_matrix(self, transistion_matrix):
        self.transistion_matrix = np.ascontiguousarray(transistion_matrix, dtype=np.float64)
        self.stationary_matrix = self.stationary_matrix_calculation(self.transistion_matrix)
        
        cl.enqueue_copy(self.queue, self.mat_buf1, self.transistion_matrix.astype(cl.cltypes.float), is_blocking=True) 
        
    def gpu_update_data(self, data):
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        cl.enqueue_copy(self.queue, self.mat_buf0, self.data.astype(cl.cltypes.float), is_blocking=True) 
        
    def gpu_timing(self, print_result=True, repeat=None):
        """
            Time the GPU execution per section, excluding the final log-likelihood calculation.
        """
        if (print_result):
            print("WARNING: Python Timing is only an estimate due to OS Scheduler.")

        t1 = time.time()
        self.gpu_calculation()
        t2 = time.time()
        if (repeat == None):
            repeat = np.round(10 / (t2 - t1)).astype(np.int) + 1

        self.timing_core = np.zeros(4)
        for i in range(repeat):
            self.gpu_calculation(timing=True)

        for i in range(4):
            self.timing_core[i] = self.timing_core[i] / repeat * 1000

        if (print_result):
            for i in range(4):
                print("E{} {} ms".format(i,self.timing_core[i]))

    def gpu_calculation(self, debug=False, timing=False):
        """
        Calculate the likelihood using the GPU and OpenCL
        """
        #t0 = time.time()
        #cl.enqueue_copy(self.queue, self.mat_buf0, self.data.astype(cl.cltypes.float), is_blocking=True)
        #cl.enqueue_copy(self.queue, self.mat_buf1, self.transistion_matrix.astype(cl.cltypes.float), is_blocking=True) 
        #cl.enqueue_copy(self.queue, self.diag_buf0, self.kernels.astype(cl.cltypes.float), is_blocking=True) 
#         n_data_dim = 2
#         n_kernel_param = 7
#         inp1 = np.zeros((self.T, n_data_dim), np.float32)
#         cl.enqueue_copy(self.queue, inp1, self.mat_buf0, is_blocking=True)
#         for i in inp1:
#             in1GPU0.append(i)
            
#         inp2 = np.zeros((dim, n_kernel_param), np.float32)
#         cl.enqueue_copy(self.queue, inp2, self.diag_buf0, is_blocking=True)
#         for i in inp2:
#             in2GPU0.append(i)

        if (timing):
            t1 = time.time()

        # Calculate diagonal probabilities from kernels and data
        completeEvent0 = cl.enqueue_nd_range_kernel(self.queue, self.kernel0, self.globalItems0, self.localItems0)
        #completeEvent0.wait()
        
        #res0 = np.zeros((T, dim),np.float32)
#         cl.enqueue_copy(self.queue, self.res0, self.res_buf0, is_blocking=True)
#         for i in self.res0:
#             outGPU0.append(i)
        if (timing):
            completeEvent0.wait()
            t2 = time.time()
            self.timing_core[0] += (t2 - t1)
            t1 = time.time()

        if (debug):
            completeEvent0.wait()
            cl.enqueue_copy(self.queue, self.res0, self.res_buf0, is_blocking=True)
            self.kernel1_output = np.copy(self.res0)
            
        # Calculate gpu diag x transistion matrix
        #   in : diagonal matrixes
        #   in : transistion matrix
        #   out: data_res -> output matrixes still in gpu memory
        completeEvent1 = cl.enqueue_nd_range_kernel(self.queue, self.kernel1, self.globalItems1, self.localItems1, wait_for=[completeEvent0])
        #completeEvent1.wait()
        
        # To check output result of step
#         cl.enqueue_copy(self.queue, self.res1, self.res_buf1, is_blocking=True) 
#         for i in self.res1:
#             outGPU1.append(i)
        if (timing):
            completeEvent1.wait()
            t2 = time.time()
            self.timing_core[1] += (t2 - t1)
            t1 = time.time()

        if (debug):
            completeEvent1.wait()
            cl.enqueue_copy(self.queue, self.res1, self.res_buf1, is_blocking=True)
            self.kernel2_output = np.copy(self.res1)
            
        # Calculate gpu matrix x matrix - 
        #   in : data -> matrix pointer
        #   out: res -> output matrixces = number of work groups 
        #   out: recCoef -> number of coefficients used by each work group
        completeEvent2 = cl.enqueue_nd_range_kernel(self.queue, self.kernel2, self.globalItems2, self.localItems2, wait_for=[completeEvent1])
        completeEvent2.wait()

        if (timing):
            completeEvent2.wait()
            t2 = time.time()
            self.timing_core[2] += (t2 - t1)
            t1 = time.time()

        cl.enqueue_copy(self.queue, self.res2, self.res_buf2, is_blocking=True)
        cl.enqueue_copy(self.queue, self.resCoef2, self.resCoef_buf2,is_blocking=True)        
        self.queue.flush()
        
#         for i in self.res2:
#             out1GPU2.append(i)

#         for i in self.resCoef2:
#             out2GPU2.append(i)



        if (debug):
            self.kernel3_output = np.copy(self.res2)
            self.kernel3_output_coeff = np.copy(self.resCoef2)
            
        MIN_PRECISION = 1e-48
        SCALE_LIMIT_MAX = 1e6
        SCALE_LIMIT_MIN = 1e-6
        # Calculate final matrix multiplication from gpu output
        res = self.res2.astype(np.float)
        R = res[0]

        Tcoef = self.resCoef2.sum()
        coef = 0
        R[ R < MIN_PRECISION] = 0
        if (np.abs(R.max()) < SCALE_LIMIT_MIN):
            coef = int(np.log2( (SCALE_LIMIT_MIN/2) / R.max() + 1 ))
            Tcoef += coef
            R = R * np.float_power(2, coef)
            coef = 0
        if (np.abs(R.max()) > SCALE_LIMIT_MAX):
            coef = int(np.log2( R.max() / (SCALE_LIMIT_MAX/2) + 1)) * -1
            Tcoef += coef
            R = R * np.float_power(2, coef)
            coef = 0
        
        for m in res[1:]:
            np.matmul(R, m, R)
            R[ R < MIN_PRECISION] = 0
            if (np.abs(R.max()) < SCALE_LIMIT_MIN):
                coef = int(np.log2( (SCALE_LIMIT_MIN/2) / np.abs(R.max()) + 1))
                Tcoef += coef
                R = R * np.float_power(2, coef)
                coef = 0
            if (np.abs(R.max()) > SCALE_LIMIT_MAX):
                coef = int(np.log2( np.abs(R.max()) / (SCALE_LIMIT_MAX/2)) + 1) * -1
                Tcoef += coef
                R = R * np.float_power(2, coef)
                coef = 0

        if (timing):
            completeEvent2.wait()
            t2 = time.time()
            self.timing_core[3] += (t2 - t1)

        R = np.dot(self.stationary_matrix, R)
        L = np.log(np.sum(R)) - np.log(2)*(Tcoef)
        #t1 = time.time()
        #print("GPU ",(t1- t0))
        if (debug):
            self.result_output = L
        return L
    
    def f_x(self, x, mean, cov):
        return sps.multivariate_normal.pdf(x[:2], mean=[0, 0], cov=[1, 1])

    def multivariate_normal(self, x, mean, cov):
        return sps.multivariate_normal.pdf(x, mean=mean, cov=cov)
        #return sps.multivariate_normal.pdf(x, mean=mean, cov=cov) / sps.multivariate_normal.pdf(mean, mean=mean, cov=cov)

    def f_zx(self,x, mean, cov, p, z): 
        """
            x - position of earthquake
            p - probability of earthquake
            z - binomial observing earthquake
        """
        if (z == 0):
            return (1-p)
        else:
            return p*self.multivariate_normal(x[:2], mean, cov)
            #return (1-p)**(1-z)*(p*self.multivariate_normal(x[:2], mean, cov))**z

    def cpu_calculation(self, timing=False):
        """
        Calculate the likelihood using the CPU
        """
        #t0 = time.time()
        minVal = 1e-6
        maxVal = 1e6

        A = np.identity(self.dim)
        B = np.identity(self.dim)
        count = 0
        count_plus = 0
        cov = np.zeros((2,2))
        temp = np.zeros(self.dim)
        for idx in range(self.T):  
            
            for k in range(self.dim):
                mean = self.kernels[k,:2]
                cov[0,0] = self.kernels[k,2]**2
                cov[0,1] = self.kernels[k,4] * self.kernels[k,2] * self.kernels[k,3]
                cov[1,0] = cov[0,1]
                cov[1,1] = self.kernels[k,3]**2
                p = self.kernels[k, 5]
                z = self.data[idx,2]
                send_data = self.data[idx]
                #print(send_data)
                temp[k] = self.f_zx(send_data, mean, cov, p, z)
                
#             outCPU0.append(np.copy(temp))
            np.matmul(np.diag(temp), self.transistion_matrix, A)
            
#            outCPU1.append(np.copy(A))
            #B = np.dot(B,A)
            np.matmul(B , A, B)
            maxCurrent = np.abs(B.max())
            if maxCurrent < minVal:
                coef = np.log2((minVal/2) / maxCurrent + 1).astype(np.int)
                B = B * np.exp2(coef)
                count += coef
            elif maxCurrent > maxVal:
                coef = np.log2(maxCurrent / (maxVal/2) + 1).astype(np.int)
                B = B * np.exp2(coef * -1)
                count_plus += coef

        L = np.log(np.sum(np.dot(self.stationary_matrix,B))) - np.log(2)*(count) + np.log(2)*(count_plus)
        #print("CPU ",(time.time() - t0))
        return L

    def forward(self, timing=False):
        """
        Calculate the likelihood using the CPU
        """
        #t0 = time.time()
        minVal = 1e-6
        maxVal = 1e6

        if (timing):
            kerneltime = 0
            matmultime = 0
            checktime = 0

        A = np.identity(self.dim)
        B = np.copy(self.stationary_matrix)
        count = 0
        count_plus = 0
        cov = np.zeros((2,2))
        temp = np.zeros(self.dim)
        for idx in range(self.T):  
            if (timing):
                point = time.time()

            for k in range(self.dim):
                mean = self.kernels[k,:2]
                cov[0,0] = self.kernels[k,2]**2
                cov[0,1] = self.kernels[k,4] * self.kernels[k,2] * self.kernels[k,3]
                cov[1,0] = cov[0,1]
                cov[1,1] = self.kernels[k,3]**2
                p = self.kernels[k, 5]
                z = self.data[idx,2]
                send_data = self.data[idx]
                #print(send_data)
                temp[k] = self.f_zx(send_data, mean, cov, p, z)
            if (timing):
                kerneltime += (time.time() - point)
                point = time.time()

#             outCPU0.append(np.copy(temp))
            np.matmul(np.diag(temp), self.transistion_matrix, A)
            
#             outCPU1.append(np.copy(A))
            #B = np.dot(B,A)
            np.matmul(B , A, B)

            if (timing):
                matmultime += (time.time() - point)
                point = time.time()

            maxCurrent = np.abs(B).max()
            if maxCurrent < minVal:
                coef = np.log2((minVal/2) / maxCurrent + 1).astype(np.int)
                B = B * np.exp2(coef)
                count += coef
            elif maxCurrent > maxVal:
                coef = np.log2(maxCurrent / (maxVal/2) + 1).astype(np.int)
                B = B * np.exp2(coef * -1)
                count_plus += coef

            if (timing):
                checktime += (time.time() - point)
                point = time.time()

        L = np.log(np.sum(B)) - np.log(2)*(count) + np.log(2)*(count_plus)

        if (timing):
            print("Kernel: ",kerneltime)
            print("Matmul: ", matmultime)
            print("Checkn: ",checktime)
        #L = np.log(np.sum(np.dot(self.stationary_matrix,B))) - np.log(2)*(count) + np.log(2)*(count_plus)
        #print("CPU ",(time.time() - t0))
        return L

    def cpu_calculation_old(self):
        """
        Calculate the likelihood using the CPU
        """
        #t0 = time.time()
        B = np.identity(self.dim)
        count = 0
        cov = np.zeros((2,2))
        temp = np.zeros(self.dim)
        for idx in range(self.T):  
            
            for k in range(self.dim):
                mean = self.kernels[k,:2]
                cov[0,0] = self.kernels[k,2]**2
                cov[0,1] = self.kernels[k,4] * self.kernels[k,2] * self.kernels[k,3]
                cov[1,0] = cov[0,1]
                cov[1,1] = self.kernels[k,3]**2
                p = self.kernels[k, 5]
                z = self.data[idx,2]
                send_data = self.data[idx]
                #print(send_data)
                temp[k] = self.f_zx(send_data, mean, cov, p, z)
                
#             outCPU0.append(np.copy(temp))
            A = np.dot(np.diag(temp), self.transistion_matrix)
            
#             outCPU1.append(np.copy(A))
            B = np.dot(B,A)
            if abs(B[int(self.dim*np.random.uniform(0,1)),int(self.dim*np.random.uniform(0,1))])<abs(1e-6):
                #print(B)
                B = 2**(6)*B
                B = B
                
                count += 1
        L = np.log(np.sum(np.dot(self.stationary_matrix,B))) - np.log(2)*(count*6)
        #print("CPU ",(time.time() - t0))
        return L
    
    def info(self):
        print("--------------------------------------------")
        print("{:30}: {}".format("Number of Data Points",self.T))
        print("{:30}: {}".format("Dimensions of Data Points",self.n_data_dim))
        print("{:30}: {}".format("Number of States", self.dim))
        print("{:30}: {}".format("Number of Kernel Parameters", self.n_kernel_param))
        print("--------------------------------------------")

    def write_test_data(self,filename):
        with open(filename,'w') as fp:
            N_DATA = self.T
            N_KERNELS = self.dim
            N_DATA_DIM = self.data.shape[1]
            N_KERNEL_PARAM = self.kernels.shape[1]
            N_COMPUTEUNITS = self.computeUnits
            N_WRKUNITS = self.wrkUnit

            fp.write("{}\n".format(N_DATA))
            fp.write("{}\n".format(N_KERNELS))
            fp.write("{}\n".format(N_DATA_DIM))
            fp.write("{}\n".format(N_KERNEL_PARAM))
            fp.write("{}\n".format(N_COMPUTEUNITS))
            fp.write("{}\n".format(N_WRKUNITS))
            for row in self.data:
                for val in row:
                    fp.write("{} ".format(val))
            fp.write("\n")
            for row in self.kernels:
                for val in row:
                    fp.write("{} ".format(val))
            fp.write("\n")    
            for row in self.transistion_matrix:
                for val in row:
                    fp.write("{} ".format(val))
            fp.write("\n")
            for val in self.stationary_matrix:
                    fp.write("{} ".format(val))
            fp.write("\n")
            for row in self.kernel1_output:
                for val in row:
                    fp.write("{} ".format(val))
            fp.write("\n")
            for mat in self.kernel2_output:
                for row in mat:
                    for val in row:
                        fp.write("{} ".format(val))
            fp.write("\n")
            for mat in self.kernel3_output:
                for row in mat:
                    for val in row:
                        fp.write("{} ".format(val))
            fp.write("\n")
            for val in self.kernel3_output_coeff:
                fp.write("{} ".format(val))
            fp.write("\n")
            fp.write("{}\n".format(self.result_output))
    # def __del__(self):
    #     print("Destruction")


# Generate Kernel
class Kernel:
    """
        Kernel with 5 parameters taking data with 3 dimensions
        Multivariate Gaussian - Two dimensions
    """
    def __init__(self, mean, cov, rho, p, N ):
        """
           mean - Mean of the Gaussian X and Y (N, 2)
           cov  - Standard Deviation of Gaussian X and Y (N, 2)
           rho  - Correlation between X and Y (N)
           p    - Probability kernel appears (N)
           N    - Number of Kernels
        """
        self.mean = mean
        self.cov = cov
        self.rho = rho
        self.p = p 
        self.N = N
    
    def __call__(self):
        K = np.ones([self.N,6])
        K[:, 0:2] = self.mean
        K[:, 2:4] = self.cov
        K[:, 4] = self.rho
        K[:, 5] = self.p
        return K
    
    def probablity_kernel_func(self):
        """
            Currently only implemented in C Code
        """
        pass



# Calculate random transistion matrix
def calc_transition_matrix(N):
    """
        Calculate random Transition Matrix of Size NxN
    """
    result = np.identity(N)
    # Add a random drift term.  We can guarantee that the diagonal terms
    #     will be larger by specifying a `high` parameter that is < 1.
    # How much larger depends on that term.  Here, it is 0.25.
    result = result + np.random.uniform(low=0., high=1, size=(N, N))
    # Lastly, divide by row-wise sum to normalize to 1.
    result = result / result.sum(axis=1, keepdims=1)
    return result

def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    
    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)

def plot_transition_matrix(P):
    plt.imshow(P, cmap=grayify_cmap('gist_earth_r'))
    plt.colorbar()
    plt.show()

def testOpenCLFunction(file_kernel, params, data, platform_id=0):    
    file_kernel = file_kernel

    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_test = abs_path + "/prob_test.cl"

    #file_test = "../../HMMLikelihood/prob_test.cl"
    
    n_kernel_param = np.array(params).shape[0]
    n_data_dim = np.array(data).shape[0]
    
    defines = "-D n_kernel_param={} -D n_data_dim={} ".format(n_kernel_param, n_data_dim) + "-cl-std=CL1.2"
    
    # Generate Code
    code_kernel = open(file_kernel, "r").read()
    code_test = open(file_test, "r").read()
    code = code_kernel + "\n" + code_test
    
    # OpenCL Initialisation
    platform = cl.get_platforms()[platform_id]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    program = cl.Program(context, code).build(defines)
    queue = cl.CommandQueue(context)    

    params_in = np.array(params).astype(cl.cltypes.float)
    data_in   = np.array(data).astype(cl.cltypes.float)
    res_out   = np.zeros(1,np.float32)
    
    # Buffer creation
    mem_flags = cl.mem_flags
    param_buf  = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=params_in)
    data_buf   = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=data_in)
    res_buf    = cl.Buffer(context, mem_flags.READ_WRITE, res_out.nbytes)

    # Assign function execution
    kernel = program.prob_test

    # Set program arguments
    globalItems = ( 1, )
    localItems = None 

    kernel.set_arg(0, param_buf )
    kernel.set_arg(1, data_buf  )
    kernel.set_arg(2, res_buf  )      
    
    completeEvent = cl.enqueue_nd_range_kernel(queue, kernel, globalItems, localItems)
    completeEvent.wait()
    
    cl.enqueue_copy(queue, res_out, res_buf, is_blocking=True)
    return res_out
