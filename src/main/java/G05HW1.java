import org.apache.spark.SparkConf;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.TaskContext;
import org.apache.spark.executor.TaskMetrics;
import org.apache.spark.memory.TaskMemoryManager;
import org.apache.spark.metrics.source.Source;
import org.apache.spark.shuffle.FetchFailedException;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.AccumulatorV2;
import org.apache.spark.util.TaskCompletionListener;
import org.apache.spark.util.TaskFailureListener;
import scala.Option;
import scala.Tuple2;
import scala.collection.Seq;

import javax.sound.midi.SysexMessage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class G05HW1 {

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: number_partitions, <path to file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> docs = sc.textFile(args[1]).repartition(K);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long numdocs, numwords;
        numdocs = docs.count();
        System.out.println("Number of documents = " + numdocs);
        JavaPairRDD<String, Long> count;

        Random randomGenerator = new Random();
        count = docs
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                    String filtered = document.replaceAll("[0-9 ]", "");
                    String[] tokens = filtered.split("[\\r\\n]+");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<Integer, Tuple2<String, Long>>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(randomGenerator.nextInt(K), new Tuple2<>(e.getKey(), e.getValue())));
                    }
                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R1)
                .flatMapToPair((triplet) -> {
                    HashMap<String, Long> counts = new HashMap<>();
                    for (Tuple2<String, Long> c : triplet._2()) {
                        counts.put(c._1(), c._2() + counts.getOrDefault(c._1(), 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()   // <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });

        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        StringBuilder o = new StringBuilder();
        for (Tuple2<String, Long> o_pairs : count.sortByKey().collect()) {
            o.append(o_pairs).append(" ");
        }
        System.out.println("Output pairs: " + o);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // IMPROVED WORD COUNT with mapPartitions
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // TODO : understand how to get pair with maximum count
        // TODO : understand how to produce the pair ("maxPartitionSize",N_max)
        count = docs
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                    String filtered = document.replaceAll("[0-9 ]", "");
                    String[] tokens = filtered.split("[\\r\\n]+");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    Long maxPartitionSize = 0L; // our max partition size
                    for (String token : tokens) {
                        counts.put(token, 1L /*+ counts.getOrDefault(token, 0L)*/);
                        maxPartitionSize++; // count 1 for each put
                    }
                    counts.put("maxPartitionSize" + TaskContext.getPartitionId(), maxPartitionSize); // insert the maxPartitionSize into the counts
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();  /// this return the iterator for the tuple
                })
                // cc = pairs.iterator (class count)
                // this reduce phase count for each worker the number of each element
                //        {Action=224, Horror=782, Drama=222, Thriller=21, Crime=210, Fantasy=212, Animation=210, Romance=220, Comedy=206, SciFi=193}
                //        {Action=196, Horror=766, Drama=203, Thriller=19, Crime=196, Fantasy=220, Animation=219, Romance=212, Comedy=231, SciFi=238}
                //        {Action=211, Horror=770, Drama=217, Thriller=16, Crime=220, Fantasy=213, Animation=201, Romance=197, Comedy=218, SciFi=237}
                //        {Action=215, Horror=737, Drama=228, Thriller=35, Crime=192, Fantasy=226, Romance=235, Animation=215, Comedy=208, SciFi=209}
                .mapPartitionsToPair((cc) -> {    // <-- REDUCE PHASE (R1)
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    while (cc.hasNext()) {
                        Tuple2<String, Long> tuple = cc.next();
                        counts.put(tuple._1(), tuple._2() + counts.getOrDefault(tuple._1(), 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()     // <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });

        long n_max = 0;
        for (int i = 0; i < K; i++) {
            for (Tuple2<String, Long> c : count.collect()) {
                if (c._1().equals("maxPartitionSize" + i)) {
                    if (c._2() >= n_max) {
                        n_max = c._2();
                    }
                }
            }
            int final_index = i;
            count.filter(w -> w._1.equals("maxPartitionSize" + final_index));
            System.out.println(count.collect());
        }

        System.out.println(count.collect());
        System.out.println("VERSION WITH SPARK PARTITIONS");
        System.out.println("Most frequent class =  pair (class, count) with max count");
        System.out.println("Max partition size = " + n_max);

    }

}