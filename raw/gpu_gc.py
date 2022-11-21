import torch
import time
import timeit
import gc
import sys


from components.model_analyzer.dcgm.dcgm_monitor import DCGMMonitor
from components.model_analyzer.tb_dcgm_types.da_exceptions import TorchBenchAnalyzerException
from components.model_analyzer.tb_dcgm_types.gpu_device_factory import GPUDeviceFactory
from components.model_analyzer.dcgm import dcgm_fields
from components.model_analyzer.dcgm.dcgm_structs import DCGMError
from components.model_analyzer.tb_dcgm_types.gpu_tensoractive import GPUTensorActive
from components.model_analyzer.tb_dcgm_types.gpu_utilization import GPUUtilization
from components.model_analyzer.tb_dcgm_types.gpu_power_usage import GPUPowerUsage
from components.model_analyzer.tb_dcgm_types.gpu_free_memory import GPUFreeMemory
from components.model_analyzer.tb_dcgm_types.gpu_used_memory import GPUUsedMemory
from components.model_analyzer.tb_dcgm_types.gpu_fp32active import GPUFP32Active
from components.model_analyzer.tb_dcgm_types.record_aggregator import RecordAggregator
from components.model_analyzer.tb_dcgm_types.tb_logger import set_logger, LOGGER_NAME
from components.model_analyzer.tb_dcgm_types.config import *
from components.model_analyzer.tb_dcgm_types.config import DEFAULT_MONITORING_INTERVAL

import logging
logger = logging.getLogger(LOGGER_NAME)
import json

class ModelAnalyzer:
    def __init__(self):
        self.gpu_factory = GPUDeviceFactory()
        self.gpus = self.gpu_factory.verify_requested_gpus(['all', ])
        # the metrics to be collected
        # self.gpu_metrics = [GPUUtilization, GPUPowerUsage,
        #                     GPUFreeMemory, GPUUsedMemory, GPUFP32Active, GPUTensorActive]
        self.gpu_metrics = [GPUFP32Active]
        # the final metric results. Its format is {GPU_UUID: {GPUUtilization: }}
        self.gpu_metric_value = {}
        self.gpu_monitor = None
        self.gpu_records = None
        self.config = AnalayzerConfig()
        set_logger()

    def start_monitor(self):
        try:
            self.gpu_monitor = DCGMMonitor(
                self.gpus, self.config.monitoring_interval, self.gpu_metrics)
            self.gpu_monitor.start_recording_metrics()
            print(">>> start_monitor = {}, {}".format(self.gpus, self.config.monitoring_interval))
            for g in self.gpus:
                print(g._device_name, g._device_uuid, g._device_id)
        except TorchBenchAnalyzerException:
            self._destory_monitor()
            raise

    def _destory_monitor(self):
        self.gpu_monitor.destroy()
        self.gpu_monitor = None
    
    def stop_monitor(self):
        """
        @return: collected gpu records @todo: add return type 
        """
        self.gpu_records = self.gpu_monitor.stop_recording_metrics()
        self._destory_monitor()
        return self.gpu_records
    
    def aggregate(self):
        gpu_record_aggregator = RecordAggregator()
        gpu_record_aggregator.insert_all(self.gpu_records)
        # @Yueming Hao todo: groupby GPU, then metric. Then calculate the flops
        records_groupby_gpu = {}
        records_groupby_gpu = gpu_record_aggregator.groupby(
            self.gpu_metrics, lambda record: record.device_uuid())
        
        for gpu in self.gpus:
            self.gpu_metric_value[gpu.device_uuid()] = {}
        for metric_type, metric in records_groupby_gpu.items():
            for gpu_uuid, metric_value in metric.items():
                self.gpu_metric_value[gpu_uuid][metric_type] = metric_value
    
    def set_monitoring_interval(self, attempted_interval):
        """
        The default monitoring internval is DEFAULT_MONITORING_INTERVAL * 1000 ms.
        """
        if attempted_interval < 0.1:
            logger.warning("The attempted interval is too short, would cause untrusted profiling results.")
        self.config.monitoring_interval = attempted_interval

    def print_flops(self):
        print("==========Summary==========")
        print(self.gpu_metric_value)
        for gpu_uuid in self.gpu_metric_value:
            gpu = self.gpu_factory.get_device_by_uuid(gpu_uuid)
            print(self.gpu_metric_value[gpu_uuid][GPUFP32Active].value())
            # TFLOPs/second = Device_SM_Count x Device_FMAs_Per_Cycle_Per_SM x 2 x Running_Frequency_KHz x DCGM_Activity / 1e+9
            print("GPU : TFLOPs/Second %.4f" % (gpu._sm_count * gpu._fma_count * 2 *
                gpu._frequency * self.gpu_metric_value[gpu_uuid][GPUFP32Active].value() / 1e+9))
        # @Yueming Hao: print all collected gpu records, for debug only
        logger.debug(json.dumps([_.to_dict() for _ in self.gpu_records], indent = 4))
    
    def calculate_flops(self, gpu_uuid=None):
        """
        The function to calculate TFLOPs/second for the desired GPU or the first available GPU.
        @return : a floating number representing TFLOPs/second.
        """
        if gpu_uuid:
            print(">>> {}".format(self.gpu_metric_value))
            if gpu_uuid in self.gpu_metric_value:
                gpu = self.gpu_factory.get_device_by_uuid(gpu_uuid)
                return gpu._sm_count * gpu._fma_count * 2 * gpu._frequency * self.gpu_metric_value[gpu_uuid][GPUFP32Active].value() / 1e+9
            else:
                raise TorchBenchAnalyzerException("No available GPU with uuid ", gpu_uuid, " found!")
        else:
            if len(self.gpu_metric_value) > 1:
                logger.warning("There are multiple available GPUs and will only return the first one's flops.")
            gpu_uuid = next(iter(self.gpu_metric_value))
            gpu = self.gpu_factory.get_device_by_uuid(gpu_uuid)
            return gpu._sm_count * gpu._fma_count * 2 * gpu._frequency * self.gpu_metric_value[gpu_uuid][GPUFP32Active].value() / 1e+9

def check_dcgm():
    try: 
        temp_model_analyzer = ModelAnalyzer()
        temp_model_analyzer.set_monitoring_interval(0.5)
        temp_model_analyzer.start_monitor()
        time.sleep(10)
        #temp_model_analyzer.print_flops()
        flops = temp_model_analyzer.calculate_flops(gpu_uuid="GPU-7ca5635c-d705-4d1d-495a-10d413cee2d7")
        print("flops = {}".format(flops))
        temp_model_analyzer.stop_monitor()
    except DCGMError as e:
        logger.error("ERROR: DCGM init failed. ", e)
        exit(-1)
    return True

check_dcgm()
sys.exit(-1)


gc.disable()

N = 100*1000*1000
M = 100
shape = (N,)
A = []
for i in range(M):
    rand_tensor = torch.rand(shape, device='cuda')
    A.append(rand_tensor)
    time.sleep(5)
    print(i, flush=True)

print("Start Sleep")
time.sleep(5)
print("Start GC")
gc.collect()
print("Done GC")
time.sleep(5)
sys.exit()

T = timeit.timeit('''
import torch
N = 1024
shape = (N,N,)
F = [1]
for i in range(100):
	rand_tensor = torch.rand(shape, device='cuda')
	F[0] = rand_tensor
	#torch.cuda.empty_cache()
	gc.collect()
	print(i)
''', 
number = 3
)
print(T)
