?$	??/?$??M???*n??F%u?k?!?V-??	?w+???@?B|N??@!????h?'@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?V-???? ?rh??A46<?R??Y?:pΈҮ?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsF%u?k?{?G?zd?A-C??6J?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??ZӼ???;?O??n??Aa2U0*?S?*	?????LX@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatz6?>W[??!??a?2pA@)?v??/??1y?5?,R=@:Preprocessing2F
Iterator::Modelj?q?????!???^B?B@)lxz?,C??1?[??"e<@:Preprocessing2U
Iterator::Model::ParallelMapV2Έ?????!??[??"#@)Έ?????1??[??"#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!{	?%??1@)Έ?????1??[??"#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?q?????!uk~X? @)?q?????1uk~X? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????߮?!Lh/??O@)a??+ey?1X?$﯃@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOv?!??8??8@)??_vOv?1??8??8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM?O???!f??#*?4@)a??+ei?1X?$﯃	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s8.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??].?2'@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	DҔ????d?ཀྵ??{?G?zd?!?? ?rh??	!       "	!       *	!       2$	[???V???LH?WOR??-C??6J?!46<?R??:	!       B	!       J	??J4[?????&'?ˡ?!?:pΈҮ?R	!       Z	??J4[?????&'?ˡ?!?:pΈҮ?JCPU_ONLYY??].?2'@b Y      Y@q:??8?@"?	
both?Your program is MODERATELY input-bound because 11.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"s8.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?31.2212% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 