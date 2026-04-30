'''
DEM calibration workflow using an MLP surrogate model and Dual Annealing.

The script iteratively calibrates microscopic DEM material parameters
against target macroscopic responses from UCS and Brazilian tensile tests.

General workflow:
1. Read previously computed DEM material-response data.
2. Normalize microscopic material parameters.
3. Compute an objective/fitness value from the difference between simulated
   and target macroscopic responses.
4. Train an MLPRegressor surrogate model on the current data.
5. Use Dual Annealing to search the surrogate surface for improved parameters.
6. Run new DEM/MUSEN simulations for the proposed parameters.
7. Append new results to the calibration database.
8. Periodically compare surrogate-calibrated results with direct MUSEN tests.
'''

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.optimize import dual_annealing
from decimal import Decimal
#import matplotlib.pyplot as plt
import asyncio
import re
import random
import subprocess, os
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D

async def run_program(command):
    """
    Run one external DEM simulation command asynchronously.

    Parameters
    ----------
    command : list[str]
        Command-line arguments used to launch RockFit-DEM or MUSEN.

    Returns
    -------
    stdout_str, stderr_str : tuple[str, str]
        Text output of the external process.
    """
    # Create the subprocess with stdout and stderr pipes
    proc = await asyncio.create_subprocess_exec( 
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )    
    # Read the standard output and standard error asynchronously
    stdout, stderr = await proc.communicate()    
    # Decode the output from bytes to string (assuming it's UTF-8 encoded)
    stdout_str = stdout.decode('utf-8')
    stderr_str = stderr.decode('utf-8')    
    # Return the output as a tuple
    return stdout_str, stderr_str

async def put_task_queue(p_Commands, p_Queue, p_flags):
    while True:
        try:
            l_free = p_flags.index(0) # Perform actions when the element is found, e.g., return index, modify the list, etc.
            l_Command = p_Commands.pop()
            print('l_Command: ',l_Command)
            p_Queue.put_nowait(run_program(l_Command))
            p_flags[l_free] = 1
        except ValueError:
            return # Perform actions when the element is not found, e.g., do something else, return a default value, etc.

async def run_tasks(p_Commands, p_NSimultaneousTasks, p_indexGPUs, p_Queue, p_TemplatePattern, p_ResultMaterialDataPoints, p_FilePath, p_WriteResult=True):
    """
    Run a batch of DEM simulations in parallel on available GPUs.

    Each command is assigned a GPU index and a unique simulation index.
    When a simulation finishes, its stdout is parsed for:
    - the simulation index,
    - the GPU that became free,
    - the calculated material-response vector.

    The parsed result is appended to ResultMaterial.txt.
    """
    #l_flags = [0] * p_NSimultaneousTasks
    l_mainindex = 0
    l_taskpattern = r'INDEX (\d+)' # Pattern used to recover the unique simulation index from stdout.
    l_TemplateGPUPattern = r'Correct GPU request! Requested:\s+(\d+)' # Pattern used to identify which GPU has finished and can receive another task.
    l_i = 0
    #l_QueueFlags = []
    l_indextasks = {}        
    #print('p_Commands: ', p_Commands)
    #while p_Commands:
        #await put_task_queue(p_Commands, p_Queue, l_flags)
    while p_Commands and l_i < p_NSimultaneousTasks:
        l_Command = p_Commands.pop()
        #l_Command[0] = l_Command[0]+'_'+str(l_i)
        l_Command[1] = 'G' + str(p_indexGPUs[l_i])
        l_Command[4] = l_Command[4] + str(l_mainindex)
        print('Command:',l_Command, ' | ', l_i, ' ', p_NSimultaneousTasks)
        #p_Queue.put_nowait(run_program(l_Command))
        task = asyncio.create_task(run_program(l_Command))
        p_Queue.append(task)
        l_indextasks[l_mainindex] = task
        #task_to_index = {task: index for index, task in enumerate(tasks)}
        #l_QueueFlags.append(False)
        #l_flags[l_i] = 1
        l_i+=1
        l_mainindex += 1
    #tasks = []
    #for i in range(p_NSimultaneousTasks):
    #    l_task = p_Queue.get()
    #    task = asyncio.create_task(l_task)
    #    print('task ', l_task, ' | ', task)
    #    tasks.append(task)
    #print('Start?')
    #exit(0)
    #await asyncio.sleep(10)
    #print('Waited ', p_Queue)
    #for p in p_Queue:
    #    print(p)
    #exit(0)
    #while not p_Queue.empty():
    #print('All ', all(l_QueueFlags))
    while p_Queue:
        #print('Queue: ', p_Queue)
        #l_done,l_pending = await asyncio.wait(p_Queue, return_when=asyncio.FIRST_COMPLETED)
        #print('Wait: ', l_done,'\n', l_pending)
        #exit(0)
        #l_indextasks = {task: index for index, task in enumerate(p_Queue)}
        #print(l_indextasks)
        #l_futuretasks = asyncio.as_completed(p_Queue)
        #print('Task ', Task)
        #exit(0)
        for l_task in asyncio.as_completed(p_Queue):
        #for l_index, l_task in enumerate(Task):        
            
            #p_Queue.remove(l_task)
            #l_task = await p_Queue.get_nowait()
            print('task: ',l_task)
            #exit(0)
            l_stdout, l_stderr = await l_task
            #print('task ', l_task)
            
            
            #l_index = l_indextasks[l_task]
            
            
            #for l_task in l_done:
            #    print(l_task.result())
            #l_stdout = l_task.result()[0]
            #l_stderr = l_task.result()[1]
            l_match = re.search(l_taskpattern, l_stdout)
            if l_match:
                l_index = int(l_match.group(1))
                #print(l_match, ' ',  l_index)
                print(l_indextasks[l_index])
                p_Queue.remove(l_indextasks[l_index])
            
            #print('INDEX: ', l_index)
            #
            #l_QueueFlags.pop(l_index)
            #p_Queue.pop(l_index)
            print("Subprocess Output:")
            print("Standard Output:")
            print(l_stdout)
            #exit(0)
            
            l_match = re.search(p_TemplatePattern, l_stdout)
            if l_match and p_WriteResult:
                
                #substring_with_template = match.group(0)  # Get the entire matched substring
                l_captured_group = l_match.group(1)  # Get the content of the first captured group
                #print(substring_with_template)  # Output: "my name is John Doe"
                #print(captured_group)  # Output: "John Doe"
                l_ResultsPoint = l_captured_group.replace(' ', '').split(',')
                l_ResultsPointData = np.array(l_ResultsPoint, dtype=float)
                p_ResultMaterialDataPoints = np.vstack((p_ResultMaterialDataPoints, l_ResultsPointData))
                #print(p_ResultMaterialDataPoints)
                with open(p_FilePath, 'a') as l_file:
                    l_file.write(l_captured_group.replace(',', ' '))
                    l_file.write('\n')
                    #for element in ResultsPoint:
                    #    file.write(element)
                    #    file.write(' ')                    
            else:
                print("Template not found.")
            print("Standard Error:")
            print(l_stderr)
            print("\n")
            l_match = re.search(l_TemplateGPUPattern, l_stdout)
            if l_match:
                l_Free = int(l_match.group(1))
                #l_indexFree = p_indexGPUs.index(l_Free)
                #l_flags[l_Free] = 0
                if p_Commands:
                    l_Command = p_Commands.pop()
                    #l_Command[0] = l_Command[0]+'_'+str(l_index)
                    l_Command[1] = 'G' + str(l_Free)
                    l_Command[4] = l_Command[4] + str(l_mainindex)
                    print('Command:',l_Command)
                    task = asyncio.create_task(run_program(l_Command))
                    p_Queue.append(task)
                    #l_QueueFlags.append(False)
                    l_indextasks[l_mainindex] = task
                    l_mainindex += 1
            break
                    #p_Queue.put_nowait(run_program(l_Command))
                    #l_flags[l_Free] = 1
                    #await put_task_queue(p_Commands, p_Queue, l_flags)
                #print('l_match', l_match.group(1))
        #exit(0)
                # Start an additional subprocess
                #if not queue.empty():
                #    additional_command = ['python', 'additional_program.py']
                #    queue.put_nowait(run_program(additional_command))
    p_Queue.clear()
    
    
async def rerun_tasks(p_Commands, p_NSimultaneousTasks, p_indexGPUs, p_Queue, p_TemplatePattern, p_ResultMaterialDataPoints, p_FilePath, p_NAllParameters, p_reU, p_reB):
    l_mainindex = 0
    l_taskpattern = r'INDEX (\d+)'
    l_TemplateGPUPattern = r'Correct GPU request! Requested:\s+(\d+)'
    l_i = 0
    l_indextasks = {}    
    while p_Commands and l_i < p_NSimultaneousTasks:
        l_Command = p_Commands.pop()
        l_Command[1] = 'G' + str(p_indexGPUs[l_i])
        l_Command[4] = l_Command[4] + str(l_mainindex)
        print('Command:',l_Command, ' | ', l_i, ' ', p_NSimultaneousTasks)
        task = asyncio.create_task(run_program(l_Command))
        p_Queue.append(task)
        l_indextasks[l_mainindex] = task        
        l_i+=1
        l_mainindex += 1
    while p_Queue:        
        for l_task in asyncio.as_completed(p_Queue):        
            l_stdout, l_stderr = await l_task            
            l_match = re.search(l_taskpattern, l_stdout)
            if l_match:
                l_index = int(l_match.group(1))
                print(l_indextasks[l_index])
                p_Queue.remove(l_indextasks[l_index])            
            print("Subprocess Output:")
            print("Standard Output:")
            print(l_stdout)
            l_match = re.search(p_TemplatePattern, l_stdout)
            if l_match:
                l_captured_group = l_match.group(1)  # Get the content of the first captured group
                l_ResultsPoint = l_captured_group.replace(' ', '').split(',')
                l_ResultsPointData = np.array(l_ResultsPoint, dtype=float)
                
                l_material = np.array(l_ResultsPointData[0:p_NAllParameters])
                # Find the index of the list where the first 6 elements match the pattern
                #print(p_ResultMaterialDataPoints[:, :p_NAllParameters], l_material)
                l_matching_material = np.all(p_ResultMaterialDataPoints[:, :p_NAllParameters] == l_material, axis = 1)
                l_matching_index = np.argmax(l_matching_material)
                if l_matching_material.any():
                    l_matching_preresult = p_ResultMaterialDataPoints[l_matching_index]
                    l_matching_result = l_matching_preresult[0:p_NAllParameters]
                    if p_reU:
                        #print(l_ResultsPointData[p_NAllParameters:p_NAllParameters+4])
                        l_matching_result = np.hstack((l_matching_result, l_ResultsPointData[p_NAllParameters:p_NAllParameters+4]))
                    else:
                        print('A\n')
                        l_matching_result = np.hstack((l_matching_result, l_matching_preresult[p_NAllParameters:p_NAllParameters+4]))
                    if p_reB:
                        l_matching_result = np.hstack((l_matching_result, l_ResultsPointData[p_NAllParameters+4:p_NAllParameters+8]))
                    else:
                        l_matching_result = np.hstack((l_matching_result, l_matching_preresult[p_NAllParameters+4:p_NAllParameters+8]))
                
                else:
                    l_matching_result = l_ResultsPointData
                print('MM ', l_matching_result)
                #exit(0)
                p_ResultMaterialDataPoints = np.vstack((p_ResultMaterialDataPoints, l_matching_result))
                with open(p_FilePath, 'a') as l_file:
                    #ls_result = np.array2string(l_matching_result, separator=' ')
                    #ls_result = ' '.join(map(str, l_matching_result))
                    ls_result = ' '.join([f"{l_e:.5e}" for l_e in l_matching_result])
                    l_file.write(ls_result)
                    l_file.write('\n')
            else:
                print("Template not found.")
            print("Standard Error:")
            print(l_stderr)
            print("\n")
            l_match = re.search(l_TemplateGPUPattern, l_stdout)
            if l_match:
                l_Free = int(l_match.group(1))                
                if p_Commands:
                    l_Command = p_Commands.pop()
                    l_Command[1] = 'G' + str(l_Free)
                    l_Command[4] = l_Command[4] + str(l_mainindex)
                    print('Command:',l_Command)
                    task = asyncio.create_task(run_program(l_Command))
                    p_Queue.append(task)
                    l_indextasks[l_mainindex] = task
                    l_mainindex += 1
            break
    p_Queue.clear()

def SetNTasks(p_Commands, p_N, p_MaterialBorderParameters, p_MaterialFixParameters, p_Command, p_NParameters, p_NFixParameters):
    """
    Generate N random DEM simulations by sampling material parameters
    uniformly within prescribed bounds.

    The generated material parameters are inserted into the command-line
    argument M(...), while the target macroscopic response is inserted into T(...).
    """
    l_material = np.zeros(p_NParameters+p_NFixParameters, dtype = np.float64)
    #print(l_material)
    for l_j in range(0,p_N):
        l_Command = p_Command.copy()
        for l_i in range(0,p_NParameters):
            l_material[l_i] = random.uniform(p_MaterialBorderParameters[l_i][0], p_MaterialBorderParameters[l_i][1])
        for l_i in range(0,p_NFixParameters):
            l_material[l_i+p_NParameters] = p_MaterialFixParameters[l_i]
        CommandStr = 'M(' + '%.8E' % Decimal(str(l_material[0])) 
        for l_i in range(1,p_NParameters+p_NFixParameters):
            CommandStr += ',' + '%.8E' % Decimal(str(l_material[l_i]))        
        l_Command[2] = CommandStr + ')'
        l_Command[3] = 'UB'
        #print('l_Command: ', l_Command)
        p_Commands.append(l_Command)
        #p_Queue.put_nowait(run_program(p_Command))
    #print(p_Commands)
    
def SetNTasksVariation(p_Commands, p_N, p_MaterialBorderParameters, p_MaterialFixParameters, p_MaterialParameters, p_Variation, p_Command, p_NParameters, p_NFixParameters):
    """
    Generate a local cloud of simulations around the current best material point.

    The first command uses the proposed material parameters directly.
    The remaining commands randomly perturb each calibrated parameter within
    ±p_Variation, clipped to the global material-parameter bounds.
    """
    l_material = np.zeros(p_NParameters+p_NFixParameters, dtype = np.float64)
    l_Command = p_Command.copy()
    for l_i in range(0,p_NParameters):
        l_material[l_i] = p_MaterialParameters[l_i]
    for l_i in range(0,p_NFixParameters):
        l_material[l_i+p_NParameters] = p_MaterialFixParameters[l_i]
    CommandStr = 'M(' + '%.8E' % Decimal(str(l_material[0])) 
    for l_i in range(1,p_NParameters+p_NFixParameters):
        CommandStr += ',' + '%.8E' % Decimal(str(l_material[l_i]))        
    l_Command[2] = CommandStr + ')'
    l_Command[3] = 'UB'
    p_Commands.append(l_Command)
            
    #print('NGPU', p_N)
    for l_j in range(1,p_N):
        l_material = np.zeros(p_NParameters+p_NFixParameters, dtype = np.float64)
        l_Command = p_Command.copy()
        for l_i in range(0,p_NParameters):
            l_border = np.clip([p_MaterialParameters[l_i]*(1.0-p_Variation),p_MaterialParameters[l_i]*(1.0+p_Variation)], p_MaterialBorderParameters[l_i][0], p_MaterialBorderParameters[l_i][1])
            #print('border ', l_border)
            l_material[l_i] = random.uniform(l_border[0], l_border[1])
            #print('l_i ', l_material[l_i])
        #print('l_material ', l_j, ' ', p_N, ' ', l_material)
        for l_i in range(0,p_NFixParameters):
            l_material[l_i+p_NParameters] = p_MaterialFixParameters[l_i]
        CommandStr = 'M(' + '%.8E' % Decimal(str(l_material[0])) 
        for l_i in range(1,p_NParameters+p_NFixParameters):
            CommandStr += ',' + '%.8E' % Decimal(str(l_material[l_i]))
        #print('CommandStr ', CommandStr)
        l_Command[2] = CommandStr + ')'
        l_Command[3] = 'UB'
        p_Commands.append(l_Command)
        #p_Queue.put_nowait(run_program(p_Command))
    #print(p_Commands)
        
# Surrogate model:
# maps normalized microscopic DEM parameters to the scalar calibration objective.
# Inputs: normalized material parameters.
# Output: normalized objective/fitness value.
mlp = MLPRegressor(
    hidden_layer_sizes=[300,500,900,500,300],
    max_iter=20000,
    tol=0,
)

def z(x,y):
    return 1.0-np.exp(-(np.square(x) + np.square(y+0.5))/0.1)

counter = 0
def Material_Surface(x):
    """
    Objective function passed to Dual Annealing.

    Instead of running a DEM simulation directly, this function evaluates
    the trained MLP surrogate at the candidate normalized parameter vector x.
    Dual Annealing therefore searches the surrogate-predicted calibration error.
    """
    global counter
    global mlp
    counter+=1
    return mlp.predict([(x[0], x[1], x[2], x[3])])
    #return 1.0-np.exp(-(np.square(x[0]) + np.square(x[1]+0.5))/0.1)    

def readfile_ResultMaterial(filename):
    with open(filename, 'r') as file:
        # Read lines from the file and split by commas to get the elements of each sublist
        content = [line.strip().split(' ') for line in file]
    # Convert elements to integers
    DataPoints = [[float(element) for element in sublist] for sublist in content]
    #print(DataPoints, '_________________________________\n')
    #print(np.array(DataPoints))
    return np.array(DataPoints)
    
def NormalizedParameters(p_Array, p_NormalizationCoefficients, p_NParameters):
    """
    Convert physical DEM parameters to normalized parameters used for ML training.

    Example:
    Young's modulus, bond strengths, and bond radius may differ by many orders
    of magnitude, so normalization prevents large-scale parameters from
    dominating the MLP training.
    """
    l_A = []
    for l_i in range(0, p_NParameters):
        l_A.append(p_Array[l_i]/p_NormalizationCoefficients[l_i])
    #l_A.append(p_Array[4])
    return l_A

def ReNormalizedParameters(p_Array, p_NormalizationCoefficients, p_NParameters):
    """
    Convert normalized optimizer output back to physical DEM parameters.
    """
    l_A = []
    for l_i in range(0, p_NParameters):
        l_A.append(p_Array[l_i]*p_NormalizationCoefficients[l_i])
    #l_A.append(p_Array[4])
    return l_A
    
def NormalizedResult(p_Array, p_NormalizationCoefficient):
    l_A = []
    for l_a in p_Array:
        l_A.append(l_a/p_NormalizationCoefficient)
    return l_A
    
def FullNormalizedResult(p_Array):
    l_min = min(p_Array)
    l_max = max(p_Array)
    if l_max-l_min>1e-9:
        l_coeff = 1.0/(l_max-l_min)
    else:
        l_coeff = 1.0
    l_A = []
    for l_a in p_Array:
        l_A.append((l_a-l_min)*l_coeff)
    return l_A
    
def FullNormalizedResult2(p_Array):
    l_min = min(p_Array)
    l_max = max(p_Array)
    if l_max>1e-9:
        l_coeff = 1.0/l_max
    else:
        l_coeff = 1.0
    l_A = []
    for l_a in p_Array:
        l_A.append(l_a*l_coeff)
    return l_A
    
def ReCalculateTargetFunctional(p_XY, p_TargetResult, p_WeightResult):
    """
    Compute the scalar calibration objective for each simulated material point.

    The objective compares simulated macroscopic quantities with target values.
    Relative errors are squared and multiplied by user-defined weights.

    Special penalty rules are applied to selected response components:
    - component 3 appears to penalize unacceptable deformation/damage behavior,
    - component 5 appears to penalize Brazilian-test-related mismatch.

    The objective is capped to avoid extremely large penalties dominating
    the training set.
    """
    Y = []
    l_i = 0
    for l_x in p_XY:
        l_y = 0
        for l_j in range(0,6):
            if l_j == 3:
                if 1.0-l_x[l_j]<p_TargetResult[l_j] or 1.0-l_x[l_j]>0.7:                    
                    l_dyt = 10
                else:
                    l_dyt = 0
            elif l_j == 5:
                if 1.0-l_x[l_j]<p_TargetResult[l_j]:                    
                    l_dyt = 10
                else:
                    l_dyt = 0
                #print('Y0', l_j, ' ', l_x[l_j], ' ', p_TargetResult[l_j], ' ', l_dyt)
            elif l_x[l_j] >= p_TargetResult[l_j]:
                #print('Y1', l_j, ' ', l_x[l_j], ' ', p_TargetResult[l_j])
                l_dyt = l_x[l_j]/p_TargetResult[l_j] - 1.0
            else:
                #print('Y2', l_j, ' ', l_x[l_j], ' ', p_TargetResult[l_j], ' | ', l_x)
                l_dyt = p_TargetResult[l_j]/l_x[l_j] - 1.0
            if l_dyt > 10:
                l_dyt = 10
            l_y += p_WeightResult[l_j]*l_dyt*l_dyt
        l_i+=1
        #print('R ', l_x, '|', l_y)
        Y.append(l_y)
    #exit(0)
    return Y
    
def GetNewArrayOfPoints(p_X, p_Y, p_Variation, p_externalVariation, p_NMinPoints, p_NParameters):
    """
    Select training points near the current best material parameter set.

    The best point is the point with the minimum objective value.
    A local parameter window is constructed around it.
    Existing points inside the wider window are retained for surrogate training.

    If too few local points are available, the function returns the number of
    additional random DEM simulations required to enrich the local database.
    """
    l_V0 = 1.0-p_Variation
    l_V1 = 1.0+p_Variation    
    l_rowindex_min = np.argmin(p_Y)
    l_material = p_X[l_rowindex_min]
    #print('l_material ', l_material, ' | ', p_Variation)
    #print('All_Y:',p_Y)
    l_border = [tuple((l_e/p_Variation, l_e*p_Variation)) for l_e in l_material]
    l_externalborder = [tuple((l_e/p_externalVariation, p_externalVariation*l_e)) for l_e in l_material]
    print('Min ', l_rowindex_min, ' ', p_Y[l_rowindex_min], ' | ', l_material, ' | ', l_border, ' | ', l_externalborder)
    l_X = []
    l_Y = []
    l_size_p_X = len(p_X)
    for l_i in range(0, l_size_p_X):
        l_e = p_X[l_i]
        l_flag = 1
        for l_j in range(0, p_NParameters):            
            if l_externalborder[l_j][0]>l_e[l_j] or l_e[l_j]>l_externalborder[l_j][1]:
                l_flag = 0            
        if l_flag==1:
            l_X.append(p_X[l_i])
            l_Y.append(p_Y[l_i])
    if p_NMinPoints - len(l_Y) > 0:
        l_NRand = p_NMinPoints - len(l_Y)
    else:
        l_NRand = 0
    #print('X:', l_X, '\nY:', l_Y, '\nRand:', l_NRand)
    return l_NRand, l_X, l_Y, l_border, l_rowindex_min

def DeleteFileLines(p_FilePath, p_LineNumbers):
    with open(p_FilePath, 'r') as file:
        l_lines = file.readlines()
        
    l_linestosave = [l_l for l_il, l_l in enumerate(l_lines) if l_il not in p_LineNumbers]
    #for l_il in LineNumbers:
    #    if 0 <= l_il <= len(l_lines)-1:
    #        del l_lines[l_il]  # Adjust for 0-based indexing
    #    else:
    #        print("Line number out of range.")
    with open(p_FilePath, 'w') as file:
        file.writelines(l_linestosave)

def ReCalculatePoints(p_Commands, p_ResultMaterialDataPoints, p_MaterialFixParameters, p_Command):
    l_linestodelete = []
    l_j = 0
    for l_R in p_ResultMaterialDataPoints:        
        if(abs(l_R[10]-1e6)>1e-10):
            print('R ',l_j, ' | ', l_R)
            l_material = np.zeros(6, dtype = np.float64)
            l_Command = p_Command.copy()
            for l_i in range(0,4):
                l_material[l_i] = l_R[l_i]
            for l_i in range(0,1):
                l_material[l_i+4] = p_MaterialFixParameters[l_i]
            CommandStr = 'M(' + '%.8E' % Decimal(str(l_material[0])) 
            for l_i in range(1,5):
                CommandStr += ',' + '%.8E' % Decimal(str(l_material[l_i]))        
            l_Command[2] = CommandStr + ')'
            #print('l_Command: ', l_Command)
            p_Commands.append(l_Command)
            l_linestodelete.append(l_j)            
        l_j+=1
        
    #print(p_Commands)
    print(l_linestodelete)
    return l_linestodelete
    
def ReCalculatePointsUB(p_Commands, p_ResultMaterialDataPoints, p_MaterialFixParameters, p_Command, p_NParameters, p_NFixParameters, p_Type):
    l_linestodelete = []
    l_j = 0
    for l_R in p_ResultMaterialDataPoints:
        if(abs(l_R[10]-1e6)>1e-10 and l_j<480):
            print('R ',l_j, ' | ', l_R)
            l_material = np.zeros(p_NParameters+p_NFixParameters, dtype = np.float64)
            l_resultU = l_R[p_NParameters+p_NFixParameters:p_NParameters+p_NFixParameters+4]
            l_resultB = l_R[p_NParameters+p_NFixParameters+4:p_NParameters+p_NFixParameters+8]
            #if p_Type.find('U'):
            if abs(l_resultB[0]-0.01)>1e-13 or abs(l_resultB[1]-1e6)>1e-13:
                #print(abs(l_resultB[0]-0.01), abs(l_resultB[1]-1e6))
                continue            
            #print(l_resultU, l_resultB)
            #exit(0)
            l_Command = p_Command.copy()
            for l_i in range(0,p_NParameters):
                l_material[l_i] = l_R[l_i]
            for l_i in range(0,p_NFixParameters):
                l_material[l_i+p_NParameters] = p_MaterialFixParameters[l_i]
            CommandStr = 'M(' + '%.8E' % Decimal(str(l_material[0])) 
            for l_i in range(1,p_NParameters+p_NFixParameters):
                CommandStr += ',' + '%.8E' % Decimal(str(l_material[l_i]))        
            l_Command[2] = CommandStr + ')'
            l_Command[3] = p_Type
            #print('l_Command: ', l_Command)
            p_Commands.append(l_Command)
            l_linestodelete.append(l_j)            
        l_j+=1
        #break
        
    #print(p_Commands)
    print(l_linestodelete)
    return l_linestodelete
    
   
materialprop=['YOUNG_MODULUS', 'NORMAL_STRENGTH', 'TANGENTIAL_STRENGTH', 'DENSITY']
   

def read_CT(p_result, i_var, j_var):
    """
    Read MUSEN compression-test output and extract peak axial response.

    The function reads displacement and force from the exported CSV file.
    It stores:
    p_result[0] = axial strain at peak force,
    p_result[1] = compressive strength estimated from peak force / specimen area.
    """
#np.float64
    nameF = './MUSEN_calc/CompressionTest_{jj}_{ii}.csv'.format(ii=i_var, jj=j_var)
    df = pd.read_csv(nameF, sep='; ', header=0, dtype=object, na_filter=True, usecols=['Z[m]', 'Z[N]'], engine='python', na_values = "-nan(ind)")
    maxF=0
    a1=0
    maxD=0
    Nf = df.shape[0]
    for i in range (0, Nf):
        a1 = float(df['Z[N]'][i])
        if a1>maxF:
            maxF=a1
            maxD=-float(df['Z[m]'][i])
    p_result[0] = 2.0*maxD/0.05
    p_result[1] = maxF/(np.pi*0.0125*0.0125)
    #print(maxD,maxF)
    return 0
    
def read_BT(p_result, i_var, j_var):
    """
    Read MUSEN Brazilian-test output and extract peak tensile-strength proxy.

    The function reads the maximum force and converts it to an equivalent
    Brazilian tensile stress using the specimen geometry.
    """
    nameF = './MUSEN_calc/BrasilTest_{jj}_{ii}.csv'.format(ii=i_var, jj=j_var)
    df = pd.read_csv(nameF, sep='; ', header=0, dtype=object, na_filter=True, usecols=['Z[m]', 'Z[N]'], engine='python', na_values = "-nan(ind)")
    maxF=0
    a1=0
    maxD=0
    Nf = df.shape[0]
    for i in range (0, Nf):
        a1 = float(df['Z[N]'][i])
        if a1>maxF:
            maxF=a1
            maxD=-float(df['Z[m]'][i])
    #yy_var[0][i_var] = maxD
    p_result[2] = maxF/(0.5*np.pi*0.025*0.025)
    #print(maxF)
    #pause()
    return 0
    
def save_CT_script(p_material, i_var, j_var):
    """
    Write a MUSEN script for a uniaxial compression test.

    The script:
    1. Generates bonds in the sample.
    2. Assigns calibrated material properties.
    3. Runs the GPU simulator.
    4. Exports force-displacement data for post-processing.
    """
#np.float64
    global materialprop
    nameF = './MUSEN_calc/ScriptCT_{jj}_{ii}.dat'.format(ii=i_var, jj=j_var)
    f = open(nameF, "w")
    f.write('JOB\n')
    f.write('SOURCE_FILE ./MUSEN_calc/CT_sample.mdem\n')
    f.write('RESULT_FILE ./MUSEN_calc/resultsCT_Bond_{jj}_{ii}.mdem\n'.format(ii=i_var, jj=j_var))
    f.write('COMPONENT BONDS_GENERATOR\n')
    f.write('PACK_GEN_MATERIAL 1 BHC3WRNUQZ\n')    
    f.write('BOND_GEN_MINDIST 1 {value}\n'.format(value=-p_material[4]))
    f.write('BOND_GEN_MAXDIST 1 {value}\n'.format(value=p_material[4]))
    f.write('BOND_GEN_DIAMETER 1 {value}\n'.format(value=2.0*p_material[3]))
    f.write('BOND_GEN_OVERLAY 1 NO\n') 
    f.write('\n')
    f.write('JOB\n')
    f.write('SOURCE_FILE ./MUSEN_calc/resultsCT_Bond_{jj}_{ii}.mdem\n'.format(ii=i_var, jj=j_var))
    f.write('RESULT_FILE ./MUSEN_calc/resultsCT_{jj}_{ii}.mdem\n'.format(ii=i_var, jj=j_var))
    f.write('COMPONENT SIMULATOR\n')
    f.write('MATERIAL_PROPERTY {name} BHC3WRNUQZ {value}\n'.format(name=materialprop[0], value=p_material[0]))
    f.write('MATERIAL_PROPERTY {name} BHC3WRNUQZ {value}\n'.format(name=materialprop[1], value=p_material[1]))
    f.write('MATERIAL_PROPERTY {name} BHC3WRNUQZ {value}\n'.format(name=materialprop[2], value=p_material[2]))
    f.write('MATERIAL_PROPERTY {name} BHC3WRNUQZ {value}\n'.format(name=materialprop[3], value=2610)) #np.pi*((0.01)*(0.01))*(0.025)*2610/((9090)*4/3*np.pi*(0.125*1e-9))
    f.write('END_TIME 15e-3\n')
    f.write('SIMULATOR_TYPE GPU\n')
    #f.write('STOP_CRITERION BROKEN_BONDS 15000\n')
    f.write('STOP_CRITERION BROKEN_BONDS 20000\n')    
    f.write('\n')
    f.write('JOB\n')
    f.write('COMPONENT RESULTS_ANALYZER\n')
    f.write('SOURCE_FILE ./MUSEN_calc/resultsCT_{jj}_{ii}.mdem\n'.format(ii=i_var, jj=j_var))
    f.write('POSTPROCESS GeometriesAnalyzer Distance,ForceTotal BoxU ./MUSEN_calc/CompressionTest_{jj}_{ii}.csv'.format(ii=i_var, jj=j_var))
    #f.write('POSTPROCESS GeometriesAnalyzer Distance,ForceTotal Geometry BoxU Filename CompressionTest_{jj}_{ii}.dat'.format(ii=i_var, jj=j_var))
    f.close()    
    return 0

def save_BT_script(p_material, i_var, j_var):
#np.float64
    
    global materialprop
    nameF = './MUSEN_calc/ScriptBT_{jj}_{ii}.dat'.format(ii=i_var, jj=j_var)
    f = open(nameF, "w")
    f.write('JOB\n')
    f.write('SOURCE_FILE ./MUSEN_calc/BT_sample.mdem\n')
    f.write('RESULT_FILE ./MUSEN_calc/resultsBT_Bond_{jj}_{ii}.mdem\n'.format(ii=i_var, jj=j_var))
    f.write('COMPONENT BONDS_GENERATOR\n')
    f.write('PACK_GEN_MATERIAL 1 BHC3WRNUQZ\n')    
    f.write('BOND_GEN_MINDIST 1 {value}\n'.format(value=-p_material[4]))
    f.write('BOND_GEN_MAXDIST 1 {value}\n'.format(value=p_material[4]))
    f.write('BOND_GEN_DIAMETER 1 {value}\n'.format(value=2.0*p_material[3]))
    f.write('BOND_GEN_OVERLAY 1 NO\n') 
    f.write('\n')
    f.write('JOB\n')
    f.write('SOURCE_FILE ./MUSEN_calc/resultsBT_Bond_{jj}_{ii}.mdem\n'.format(ii=i_var, jj=j_var))
    f.write('RESULT_FILE ./MUSEN_calc/resultsBT_{jj}_{ii}.mdem\n'.format(ii=i_var, jj=j_var))
    f.write('COMPONENT SIMULATOR\n')
    f.write('MATERIAL_PROPERTY {name} BHC3WRNUQZ {value}\n'.format(name=materialprop[0], value=p_material[0]))
    f.write('MATERIAL_PROPERTY {name} BHC3WRNUQZ {value}\n'.format(name=materialprop[1], value=p_material[1]))
    f.write('MATERIAL_PROPERTY {name} BHC3WRNUQZ {value}\n'.format(name=materialprop[2], value=p_material[2]))
    f.write('MATERIAL_PROPERTY {name} BHC3WRNUQZ {value}\n'.format(name=materialprop[3], value=2610))
    f.write('END_TIME 15e-3\n')
    f.write('SIMULATOR_TYPE GPU\n')
    #f.write('STOP_CRITERION BROKEN_BONDS 9000\n')
    f.write('STOP_CRITERION BROKEN_BONDS 20000\n')
    
    f.write('\n')
    f.write('JOB\n')
    f.write('COMPONENT RESULTS_ANALYZER\n')
    f.write('SOURCE_FILE ./MUSEN_calc/resultsBT_{jj}_{ii}.mdem\n'.format(ii=i_var, jj=j_var))
    f.write('POSTPROCESS GeometriesAnalyzer Distance,ForceTotal BoxU ./MUSEN_calc/BrasilTest_{jj}_{ii}.csv'.format(ii=i_var, jj=j_var))
    #f.write('POSTPROCESS GeometriesAnalyzer Distance,ForceTotal Geometry BoxU Filename CompressionTest_{jj}_{ii}.dat'.format(ii=i_var, jj=j_var))
    f.close()    
    return 0

def check_if_samples_created():
    SampleU_path = './result/Sample_0_0.txt'
    SampleB_path = './result/Sample_1_0.txt'
    result = ''
    # Check if the file exists
    if not os.path.exists(SampleU_path):
        result = result + 'FUSC '
    if not os.path.exists(SampleB_path):
        result = result + 'FBSC '
    return result

async def main():
    FilePath = './result/ResultMaterial.txt'
    FileMusenPath = './result/ResultMusenMaterial.txt'
    TemplateCommand = ['./CUDARuntime4.exe', 'G0', 'M(1,2,3,4,5,6)', '', 'I']#'CS'
    Commands = []
    TemplatePattern = r"ResultCalculateOneSample\[(.*?)\]"
    NParameters = 4
    #TargetResult = [0.00480002*(0.0158726/0.0132), 104.0497482e6*(281.709/311.21), 1.0, 0.1, 10.5e6, 0.05]
    # Target macroscopic response for chalk from papers - UCT strain, UCT stress, ..., ..., BT stress
    TargetResult = np.array([0.00480002, 104.0497482e6, 1.0, 0.1, 10.5e6, 0.05]) #(0.00356305/0.0056), (72.4896/175.412),(6.02892/17.113)
    WeightResult = [1, 1, 5000, 1, 1, 1] #[50, 100, 10000, 1]
    CoeffToMusen = np.array([0.60354167, 0.33116822, 1.0, 1.0, 0.32046037, 1.0])#[4.48848e+11 4.39594e+08 9.73767e+08 7.56476e-05 5.00000e-04 3.90000e-01]  |  [4.80000000e-03 1.10428168e+08 1.02182057e+07]
    MusenResult = np.zeros(3, dtype = np.float64)
    Variation = 1.0 + 0.30
    MaterialBorderParameters = [(1.5e+10, 3.5e11), (1e7, 1e11), (1e7, 1e11), (1e-6,0.9e-3)]
    MaterialFixParameters = [5e-4, 0.39] # Fixed DEM parameters that are not optimized in this calibration run. These are appended to the calibrated parameters when creating simulations.
    NFixParameters = 2
    NParamAll = NParameters+NFixParameters
    NormalizationCoefficients = [2e11, 1e9, 1e9, 5e-4, 20]
    nMaterialBorderParameters = []
    for l_i in range(0,NParameters):
        nMaterialBorderParameters.append(tuple([MaterialBorderParameters[l_i][0]/NormalizationCoefficients[l_i], MaterialBorderParameters[l_i][1]/NormalizationCoefficients[l_i]]))
    #print(nMaterialBorderParameters)   
    #exit(0)    
    NTry = 5000
    NGPUs = 1
    indexGPUs = [0,1,2,4]
    NPrePoints = 0*10*NGPUs
    Queue = []
    #ResultMaterialDataPoints = np.empty((11,))    
    ResultMaterialDataPoints = readfile_ResultMaterial(FilePath)
    
    
    l_fsc = check_if_samples_created()
    print('l_fsc', l_fsc)
    if len(l_fsc) > 0:
        x0_train = [tuple(NormalizedParameters(e[:NParameters], NormalizationCoefficients, NParameters)) for e in ResultMaterialDataPoints]
        xy0_train = [np.concatenate((e[NParamAll:NParamAll+4],e[NParamAll+5:NParamAll+6],e[NParamAll+7:NParamAll+8])) for e in ResultMaterialDataPoints]
        CorrectedTargetResult = TargetResult*CoeffToMusen
        print('CorrectedTargetResult ', l_i, ' | ', CorrectedTargetResult)
        print('CoeffToMusen ', CoeffToMusen)
        y0_train = NormalizedResult(ReCalculateTargetFunctional(xy0_train, CorrectedTargetResult, WeightResult), NormalizationCoefficients[NParameters])
        NAPoints, x_train, y_train, x_border, iBestParameters = GetNewArrayOfPoints(x0_train, y0_train, Variation, Variation, 50, NParameters)
        #print('M ', ResultMaterialDataPoints[iBestParameters][0:7])        
        l_material = ResultMaterialDataPoints[iBestParameters][0:7]
        l_Command = TemplateCommand.copy()            
        CommandStr = 'M(' + '%.8E' % Decimal(str(l_material[0])) 
        for l_i in range(1,6):
            CommandStr += ',' + '%.8E' % Decimal(str(l_material[l_i]))        
        l_Command[2] = CommandStr + ')'
        l_Command[3] = l_fsc
        print('l_Command: ', l_Command)
        Commands.append(l_Command)
        await run_tasks(Commands, NGPUs, indexGPUs, Queue, TemplatePattern, ResultMaterialDataPoints, FilePath, p_WriteResult = False)
        #exit(0)
    #ReCalculatePointsUB(Commands, ResultMaterialDataPoints, MaterialFixParameters, TemplateCommand, NParameters, NFixParameters, 'B')
    #await run_tasks(Commands, NGPUs, indexGPUs, Queue, TemplatePattern, ResultMaterialDataPoints, FilePath)
    #await rerun_tasks(Commands, NGPUs, indexGPUs, Queue, TemplatePattern, ResultMaterialDataPoints, FilePath, NParameters+NFixParameters, False, True)
    #exit(0);
    #LinesToDelete = ReCalculatePoints(Commands, ResultMaterialDataPoints, MaterialFixParameters, TemplateCommand)
    #DeleteFileLines(FilePath, LinesToDelete)
    #print('C ', Commands)
    #await run_tasks(Commands, NGPUs, indexGPUs, Queue, TemplatePattern, ResultMaterialDataPoints, FilePath)
    #exit(0)
    #print(ResultMaterialDataPoints, '_______________________________\n')
    #exit(0)
    #result = asyncio.run(run_program(command1))
    #Queue = asyncio.Queue()
    
    # Specify the file path
    
    
    if NPrePoints > 0 :
        SetNTasks(Commands, NPrePoints, MaterialBorderParameters, MaterialFixParameters, TemplateCommand, NParameters, NFixParameters)
        await run_tasks(Commands, NGPUs, indexGPUs, Queue, TemplatePattern, ResultMaterialDataPoints, FilePath)
    #Material = ResultMaterialDataPoints[1,:5]
    #print('Material: ', Material)
    #SetNTasksVariation(Commands, NGPUs, MaterialBorderParameters, MaterialFixParameters, Material, 0.2, TemplateCommand)
    #exit(0)
    #exit(0)
    #for command in commands:
    #queue.put_nowait(run_program(command))
    #queue.put_nowait(run_program(command))
    l_musenstep = 0
    for l_i in range(0,NTry):
        x0_train = [tuple(NormalizedParameters(e[:NParameters], NormalizationCoefficients, NParameters)) for e in ResultMaterialDataPoints] # 1. Build the current training dataset from all available DEM results.
        xy0_train = [np.concatenate((e[NParamAll:NParamAll+4],e[NParamAll+5:NParamAll+6],e[NParamAll+7:NParamAll+8])) for e in ResultMaterialDataPoints]
        #print(xy0_train, "______________________________________")
        CorrectedTargetResult = TargetResult*CoeffToMusen
        print('CorrectedTargetResult ', l_i, ' | ', CorrectedTargetResult)
        print('CoeffToMusen ', CoeffToMusen)
        y0_train = NormalizedResult(ReCalculateTargetFunctional(xy0_train, CorrectedTargetResult, WeightResult), NormalizationCoefficients[NParameters]) # 2. Convert simulated macroscopic responses into scalar objective values.
        #print(x0_train, "______________________________________")
        
        NAPoints, x_train, y_train, x_border, iBestParameters = GetNewArrayOfPoints(x0_train, y0_train, Variation, Variation, 50, NParameters) # 3. Select points near the current best solution for local surrogate training.
        #print('x_border', x_border)
        #print('xy0_train', len(x_train), len(y_train))
        #exit(0)
        print('NA', NAPoints)
        if NAPoints > 0 and l_i > 0: # 4. If the local region has too few data points, run extra DEM simulations.
            ren_x_border = []
            for l_i in range(0,NParameters):
                ren_x_border.append(tuple([x_border[l_i][0]*NormalizationCoefficients[l_i], x_border[l_i][1]*NormalizationCoefficients[l_i]]))
            #print('ren_x_border', ren_x_border)
            #exit(0)
            SetNTasks(Commands, NAPoints, ren_x_border, MaterialFixParameters, TemplateCommand, NParameters, NFixParameters)
            
            await run_tasks(Commands, NGPUs, indexGPUs, Queue, TemplatePattern, ResultMaterialDataPoints, FilePath)            
            continue
        y_train = FullNormalizedResult2(y_train)
        #print('xy0_train', len(x_train), y_train)
        #exit(0)
        #y_train = NormalizedResult(ResultMaterialDataPoints[:,10], NormalizationCoefficients[4])
        #print('XY:',xy_train)
        #print('X:',x_train)
        #print('Y:',y_train)  
        #exit(0)
        mlp.fit(x_train,y_train) # 5. Train MLP on the local normalized dataset.
        ret = dual_annealing(Material_Surface, x_border) # 6. Optimize the surrogate objective using Dual Annealing.
        Material = ReNormalizedParameters(ret.x, NormalizationCoefficients, NParameters) # 7. Convert normalized optimized parameters back to physical DEM units.
        print('ret ',ret.x, ' | ', ret.fun, ' | ', counter, ' | ', Material)
        SetNTasksVariation(Commands, NGPUs, MaterialBorderParameters, MaterialFixParameters, Material, 0.2, TemplateCommand, NParameters, NFixParameters) # 8. Run new DEM simulations around the proposed optimum.
        #exit(0)
        await run_tasks(Commands, NGPUs, indexGPUs, Queue, TemplatePattern, ResultMaterialDataPoints, FilePath)
        #exit(0)
        if l_i % 4 == 0:            
            MaterialMusen = ResultMaterialDataPoints[iBestParameters][0:6]            
            save_CT_script(MaterialMusen, 0, l_musenstep)
            print('MaterialMusen ', iBestParameters, ' | ', MaterialMusen)
            nameS = 'C:/Program Files/MUSEN/CMUSEN.exe'#"/home/artem/Projects/Drill/Drill_0.0038628_linux/MUSEN_calc/ScriptCT_{jj}_{ii}.dat"
            argS ='-s=./MUSEN_calc/ScriptCT_{jj}_{ii}.dat'.format(ii=0, jj=l_musenstep)
            subprocess.run([nameS] + [argS], check=True)#
            read_CT(MusenResult, 0, l_musenstep)            
            save_BT_script(MaterialMusen, 0, l_musenstep)
            nameS = "C:/Program Files/MUSEN/CMUSEN.exe"
            argS ="-s=./MUSEN_calc/ScriptBT_{jj}_{ii}.dat".format(ii=0, jj=l_musenstep)
            subprocess.run([nameS] + [argS], check=True)
            read_BT(MusenResult, 0, l_musenstep)
            print('Musen result ', MaterialMusen, ' | ', MusenResult)
            with open(FileMusenPath, 'a') as l_file:
                #ls_result = np.array2string(l_matching_result, separator=' ')
                #ls_result = ' '.join(map(str, l_matching_result))
                l_MaterialMusen = MaterialMusen;
                l_MaterialMusen[3] = l_MaterialMusen[3]*2.0;
                ls_material = ' '.join([f"{l_e:.5e}" for l_e in l_MaterialMusen])
                ls_result = ' '.join([f"{l_e:.5e}" for l_e in MusenResult])
                l_file.write(ls_material)
                l_file.write(' ')                
                l_file.write(ls_result)
                l_file.write('\n')
            CoeffToMusen[0] = ResultMaterialDataPoints[iBestParameters][6]/MusenResult[0]
            CoeffToMusen[1] = ResultMaterialDataPoints[iBestParameters][7]/MusenResult[1]
            CoeffToMusen[4] = ResultMaterialDataPoints[iBestParameters][11]/MusenResult[2]
            l_musenstep+=1
        #if l_i == 1:    
        #    exit(0)
        
        x_train = []
        y_train = []
        
    print('Calculation completed!')
    exit(0)
    
    
    x = np.arange(-1,1,0.5)
    xy = [(j,k) for j in x for k in x]
    out = [z(p[0],p[1]) for p in xy]
    exit(0)
    x_train, x_test, y_train, y_test = train_test_split(xy, out)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax = fig.gca(projection='3d')

    # plot train data points
    x1_vals = np.array([p[0] for p in x_train])
    x2_vals = np.array([p[1] for p in x_train])

    ax.scatter(x1_vals, x2_vals, y_train)

    # plot test data points
    x1_vals = np.array([p[0] for p in x_test])
    x2_vals = np.array([p[1] for p in x_test])

    #ax.scatter(x1_vals, x2_vals, y_test, marker='x')

    #plt.show()

    #x_train = x_train.reshape(-1,1)
    #x_test = x_test.reshape(-1,1)

    #mlp = MLPRegressor(
    #    hidden_layer_sizes=[20,50,20],
    #    max_iter=1000,
    #    tol=0,
    #)
    print(x_train)
    print(y_train)
    mlp.fit(x_train,y_train)

    predictions = mlp.predict(x_test)

    mse = mean_squared_error(y_test, predictions)

    ax.scatter(x1_vals, x2_vals, predictions, c='red')
    plt.show()

    print('MSE ', mse)
    lw = [-5.12] * 10
    up = [5.12] * 10
    #bounds=list(zip(lw, up))
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    print(bounds)
    ret = dual_annealing(z1, bounds)
    print(ret.x, ' | ', ret.fun, ' | ', counter)

if __name__ == "__main__":
    asyncio.run(main())