?$	?c?2????Ȧ.?-????~j?t?x?!?	h"lx??	D~ͭ??@}B?1&@!?ZB?83@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?	h"lx???]K?=??A?G?z???Y'1?Z??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?~j?t?x??J?4q?A??H?}]?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?<,Ԛ???g??j+???A?????g?*	gffff?Y@2F
Iterator::ModelZd;?O???!?????NF@)/n????1ۂG A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2U0*???!??fųt>@)	?c???1?[9?[?9@:Preprocessing2U
Iterator::Model::ParallelMapV2??_vO??!;?7)V?$@)??_vO??1;?7)V?$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate2??%䃎?!s??k?,@)?? ?rh??1?MA?$} @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?46<??!p-/.
?K@)vq?-??1Mf|?\?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[0]::TensorSlice-C??6z?!v??
??@)-C??6z?1v??
??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ZӼ?t?!?ô?_?@)??ZӼ?t?1?ô?_?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??y?):??!Ĵ?_?C1@)?????g?1S??鞀@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 18.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s9.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9tyΫ=O2@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?6?i???|?+?Y???J?4q?!?]K?=??	!       "	!       *	!       2$	???????̌%?????H?}]?!?G?z???:	!       B	!       J	?A`??"????e???!'1?Z??R	!       Z	?A`??"????e???!'1?Z??JCPU_ONLYYtyΫ=O2@b Y      Y@q??{?gD@"?	
both?Your program is MODERATELY input-bound because 18.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"s9.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?40.811% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 