?#$	&????????V?y"???~j?t?h?!-??????	??y=?@B4CՂ?1@!??y=?H@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$-????????^??A/?$???Y)??0???"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails"??u??q??????g?AǺ???V?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?q????o?-C??6j?AǺ???F?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails;?O??nr????_vOn?A-C??6J?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/n??r????_vOn?AǺ???F?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsX9??v????<,Ԛ???A??H?}M?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?~j?t?h?HP?s?b?AǺ???F?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???_vOn?a??+ei?Aa2U0*?C?*	???????@2F
Iterator::Model+????!PU?/F@)??"??~??1-?_?
?E@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??q????!iV=?M?A@)q???h??1a??A?@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate4??@????!	???#@)???????1q?V??#@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?%䃞???!?q??3?2@)??s????1?V?p @:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?V-??!??1*???)8gDio??1? np???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??_vO??!?&??2??)????Mb??1U?B?(???:Preprocessing2U
Iterator::Model::ParallelMapV2?I+???!??M?;???)?I+???1??M?;???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?>W[????!?2?r]4@)/n????1????_???:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchlxz?,C|?!c?̶A??)lxz?,C|?1c?̶A??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???v?!?j??????)Ǻ???v?1?j??????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice	?^)?p?!$ܝ?????)	?^)?p?1$ܝ?????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range_?Q?k?!{?????)_?Q?k?1{?????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateS?!?uq{?!?_q厔??)??_?Le?1?DCg???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor????MbP?!U?B?(???)????MbP?1U?B?(???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 46.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t49.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?z?ZOYG@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?9#J{?????:'??HP?s?b?!??^??	!       "	!       *	!       2$	R?!?uqk?a??4?:}?a2U0*?C?!/?$???:	!       B	!       J	)??0???W??WM??!)??0???R	!       Z	)??0???W??WM??!)??0???JCPU_ONLYY?z?ZOYG@b Y      Y@q???Ki0:@"?	
host?Your program is HIGHLY input-bound because 46.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t49.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?26.1891% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 