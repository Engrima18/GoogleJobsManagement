import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
from random import choice 

def preprocessing(data):
    data['Arrival_Time'] = (data['Arrival_Time']-data['Arrival_Time'].min())/1e6
    mu = 0.1                                                                    
    data['service_time'] = data["CPU"]/mu

    # jobs and their complexity in terms of number of tasks and total service time
    job_data = data.groupby('Job_ID').agg(
        Total_service_time=('service_time', 'sum'),
        Total_CPU = ('CPU','sum'),
        N_tasks=('Task_ID', 'count')
    ).reset_index()
    # remove ghost jobs (with service time = 0)
    ghost_jobs = job_data[job_data.Total_service_time == 0].Job_ID
    data = data[~data['Job_ID'].isin(ghost_jobs.to_list())]

    return data

data = pd.read_csv('Cell_a.csv')
data = preprocessing(data)
data.head()

class Task:
    def __init__(self, job, task_id, arrival_time, memory_required, service_time):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.memory_required = memory_required
        self.job_id = job.job_id
        self.job = job # object of type Job
        self.remaining_time = service_time
        self.completition_time = service_time        

class Job:
    def __init__(self, jid):
        self.tasks = []                     # list with all tasks of the job 
        self.job_id = jid
        self.completition_time = -1         # will be the max completition time of the tasks of the job
        self.arrival_time = None            # arrival time of 1 of the tasks
        self.service_time = None            # total service time of all tasks of the specific job
        self.remaining_service_time = None  # remaining service time after processing part of the tasks of this job
        self.finished = False               # flag to report if a job is finished
    
    def add_tasks(self, tasks):
        self.tasks = tasks # add tasks all toghether in one time
        self.arrival_time = self.tasks[0].arrival_time
        self.service_time = sum([task.service_time for task in self.tasks]) # we never update it
        self.remaining_service_time = self.service_time                      # init remainig time of job


    def update_job(self, finished_task):
        # remove the finished task
        self.tasks.remove(finished_task)
        # update completition time if more recent
        if finished_task.completition_time > self.completition_time:
            self.completition_time = finished_task.completition_time
        if self.tasks == []:
            self.finished = True

        
    def compute_slowdown(self):
        return self.completition_time/self.service_time
    


class Server:
    def __init__(self, server_id, memory_capacity=1):
        self.server_id = server_id
        self.memory_capacity = memory_capacity
        self.memory_usage = 0       # for process sharing
        self.queue = []             # Priority queue to store tasks based on service time  
        self.unfinished_work = 0    # total quantity of unfinished work (in the server queue)
        self.tasks_count = 0
        self.processed_tasks = 0
        self.used_resources = 0
        self.received_messages = 0
    
    def add_task(self, task):
        self.queue.append(task)      # add task to the queue
        self.tasks_count += 1        
        # self.received_messages += 1

        if len(self.queue) > 1:
            task.remaining_time += self.queue[-2].remaining_time
            # print(f"queue:{len(self.queue)}, rt:{task.remaining_time}")
        
        self.unfinished_work = task.remaining_time
        task.completition_time = task.remaining_time
        

    def remove_task(self, task):
        self.queue.remove(task) 
        self.used_resources += task.service_time
        self.processed_tasks += 1

    def __compute_arrival_rate(self, period):
        return self.tasks_count/period
    
    def __compute_mean_service_time(self):
        if self.processed_tasks == 0:
            return 0
        return self.used_resources/self.processed_tasks
    
    def compute_uc(self, period):
        return self.__compute_arrival_rate(period) * self.__compute_mean_service_time()

    
    def FCFS(self,  new_arrival, last_arrival):
        finished_tasks = []
        for task in self.queue:
            task.remaining_time = max(0,task.remaining_time - (new_arrival - last_arrival))
            # if task is finished
            if task.remaining_time == 0:
                task.job.update_job(task)
                self.remove_task(task)
            # if task is processed (in part)
            elif (task.remaining_time < task.service_time) and  (task.remaining_time > 0):
                processed = task.service_time - task.remaining_time 
                task.job.remaining_service_time -= processed

    
        self.unfinished_work = 0 if self.queue==[] else self.queue[-1].remaining_time


    def SRPT(self, new_arrival, last_arrival):
        
        finished_tasks = []
        for i, task in enumerate(self.queue):
            # evaluate the remainig time when I'm looking at the queue
            if i == 0:
                task.remaining_time = task.service_time
            else:
                task.remaining_time = task.service_time + self.queue[i-1].remaining_time
            # save the ramainig time when I look at it before update
            # this value will be the quantity that I want to remove at this interval of time
            old_remaining = task.remaining_time # old remaining time
            # update the remaining time
            task.remaining_time = max(0,old_remaining - (new_arrival - last_arrival))
            # if the task is finished remove all the old remaining time from remaining service time of the job
            if task.remaining_time == 0:
                remainder = (new_arrival - last_arrival) - old_remaining
                task.completition_time  = (new_arrival - task.arrival_time) - remainder # the complete time the task passed in the server
                task.job.remaining_service_time -= old_remaining 
                # update the job completition time
                task.job.update_job(task)
                finished_tasks.append(task)
            # if task is processed (in part)
            elif (task.remaining_time < task.service_time) and  (task.remaining_time > 0):
                task.job.remaining_service_time -= task.service_time - task.remaining_time
            
        # remove all the completed tasks from the server
        self.remove_tasks(finished_tasks)

        # print(f"unfinished work:{self.unfinished_work}")
        self.queue = sorted(self.queue, key = lambda task: task.job.remaining_service_time)       

class Dispatcher:
    def __init__(self, servers: list):
        self.all_servers = servers        # all servers
        self.total_messages = 0

       
    def LWL(self,tasks:list):
        # if with_priority:
        #     tasks = sorted(tasks, key=lambda x: x.service_time)
        for task in tasks:
            self.total_messages += len(self.all_servers) # a message from each server
            # Assuming self.all_servers is a list of Server objects
            min_servers = [server for server in self.all_servers if server.unfinished_work == min(self.all_servers, key=lambda server: server.unfinished_work).unfinished_work]
            selected_server = choice(min_servers) # select a server with min unfinished work
            # selected_server = min(self.all_servers, key=lambda server: server.unfinished_work) 
            selected_server.add_task(task)                                                     # add the task to the queue of the selected server

    def JSQ(self,tasks:list):
        for task in tasks:
            self.total_messages += len(self.all_servers)                                  # a message from each server
            selected_server = min(self.all_servers, key=lambda server: len(server.queue)) # select a server with min queue
            selected_server.add_task(task)                                                # add the task to the queue of the selected server

        
def calculate_mean_message_load(dispatcher: Dispatcher, servers: list, total_tasks: int):
    total_messages = dispatcher.total_messages + sum(server.received_messages for server in servers)
    mean_message_load = total_messages / total_tasks
    return mean_message_load

def baseline(data):
    # take all the arrival times (a list)
    arr_times = data["Arrival_Time"].unique()
    # init our servers
    servers = [Server(i) for i in range(0, 64)]
    # init dispatcher
    dispatcher = Dispatcher(servers)
    # init the dict of jobs
    jobs = {jid: Job(jid) for jid in list(data.Job_ID.unique())}
    # loop over the arrival times (our timestamps)
    grouped_data = data.groupby("Arrival_Time")
    for i, arrival in enumerate(arr_times):
        if i%10000==0:
            print(i)
        # take all tasks at that arrival time (a mini dataframe)
        arrived_tasks = data[data.Arrival_Time == arrival]

        # take the job id for the actual flow of tasks
        jid = arrived_tasks.head(1).Job_ID.values[0]
        # the key of the dict is the job ID and the value is the realtive object
        job = jobs[jid]

        # craete a list of Task objects
        tasks = []
        for jid, tid, arrival_time, cpu, memory, service_time in arrived_tasks.values:
            # init a task object
            tasks.append(Task(job, int(tid), float(arrival_time), float(memory), float(service_time)))
        # add these tasks to the Job object 
        job.add_tasks(tasks)
         
        # send all the arrived tasks from dispatcher to servers
        dispatcher.LWL(tasks)
        
        # print(f"arrival:{arrival}")
        # execute tasks on each server
        for server in servers:
            # print(f"server:{i}")
            if i == 0:
                server.FCFS(arrival, arrival) # evaluate the remaing time for each task at this server
            else:
                server.FCFS(arrival, arr_times[i-1]) # evaluate the remaing time for each task at this server

    # take completiotion time (i.e. response time) for each job
    # period = np.array([job.completition_time + job.arrival_time for job in list(jobs.values()) if job.finished])
    period = arr_times[-1]
    ct = np.array([job.completition_time for job in list(jobs.values()) if job.finished])
    # print([job.compute_slowdown() for job in list(jobs.values()) if job.tasks == []])
    # define the experiment period
    # period = max(period)     # + arr_times[-1]
    # print("period:",period)
    s = [job.compute_slowdown() for job in list(jobs.values()) if job.finished]
    uc = [server.compute_uc(period) for server in servers]
    # compute the requested metrics
    mean_results = {
        "E_R" : np.mean(ct),
        "E_S": np.mean(s),
        "rho": np.mean(uc),
        "messaging_load": calculate_mean_message_load(dispatcher, servers, data.shape[0])
    }

    r_s = pd.DataFrame({
        "R" : ct,
        "S": s
    })

    rho = pd.DataFrame({
        "rho_k": uc
    })
    #for i, server in enumerate(servers):
        #print(f"server{i}: \n {[(task.task_id, task.job_id) for task in server.queue]} , \n {server.unfinished_work}")
    return(r_s, mean_results, rho)


def SRPT_LWL(data):
    # take all the arrival times (a list)
    arr_times = data["Arrival_Time"].unique()
    # init our servers
    servers = [Server(i) for i in range(0, 64)]
    # init dispatcher
    dispatcher = Dispatcher(servers)
    # init the dict of jobs
    jobs = {jid: Job(jid) for jid in list(data.Job_ID.unique())}
    # loop over the arrival times (our timestamps)
    for i, arrival in enumerate(arr_times):
        # take all tasks at that arrival time (a mini dataframe)
        arrived_tasks = data[data.Arrival_Time == arrival]

        # take the job id for the actual flow of tasks
        jid = arrived_tasks.head(1).Job_ID.values[0]
        # the key of the dict is the job ID and the value is the realtive object
        job = jobs[jid]

        # craete a list of Task objects
        tasks = []
        for jid, tid, arrival_time, cpu, memory, service_time in arrived_tasks.values:
            # init a task object and append it to the list of arrived tasks
            tasks.append(Task(job, int(tid), float(arrival_time), float(memory), float(service_time)))
        # add these tasks to the Job object 
        job.add_tasks(tasks)
         
        # send all the arrived tasks from dispatcher to servers
        dispatcher.LWL(tasks)
        
        # print(f"arrival:{arrival}")
        # execute tasks on each server
        for server in servers:
            if i == 0:
                server.SRPT(arrival, arrival) # evaluate the remaing time for each task at this server
            else:
                server.SRPT(arrival, arr_times[i-1]) # evaluate the remaing time for each task at this server


    # take completiotion time (i.e. response time) for each job
    ct = np.array([job.completition_time for job in list(jobs.values()) if job.finished == True])
    # print([job.compute_slowdown() for job in list(jobs.values()) if job.tasks == []])
    # define the experiment period
    period = arr_times[-1]
    
    s = [job.compute_slowdown() for job in list(jobs.values()) if job.finished == True]
    uc = [server.compute_uc(period) for server in servers]
    # compute the requested metrics
    mean_results = {
        "E_R" : np.mean(ct),
        "E_S": np.mean(s),
        "rho": np.mean(uc),
        "messaging_load": calculate_mean_message_load(dispatcher, servers, data.shape[0])
    }

    r_s = pd.DataFrame({
        "R" : ct,
        "S": s
    })

    rho = pd.DataFrame({
        "rho_k": uc
    })
    # for i, server in enumerate(servers):
    #     print(f"server{i}: \n {[(task.task_id, task.job_id) for task in server.queue]} , {server.unfinished_work}")
    return(r_s, mean_results, rho)
