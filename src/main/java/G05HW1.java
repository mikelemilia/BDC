import org.apache.spark.SparkConf;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

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

        Map<String, Long> map = count.sortByKey().collectAsMap();

        // Computing the maxPartitionSize
        Long n_max = 0L;
        for (int i = 0; i < K; i++) {
            for (Tuple2<String, Long> c : count.sortByKey().collect()) {
                if (c._1().equals("maxPartitionSize" + i)) {
                    if (c._2() >= n_max) {
                        n_max = c._2();
                    }
                }
            }

            map.remove("maxPartitionSize" + i);

        }

        // todo fixare sta cazzo di stringa
        ArrayList<String> max_pair = computeMax(map);

        System.out.println("VERSION WITH SPARK PARTITIONS");
        System.out.println("Most frequent class = " + max_pair.get(0));
        System.out.println("Max partition size = " + n_max);

    }

    public static ArrayList<String> computeMax(Map<String, Long> map) {
        ArrayList<String> s = new ArrayList<>();
        Map.Entry<String, Long> maxEntry = Collections.max(map.entrySet(), Map.Entry.comparingByValue());
        for (Map.Entry<String, Long> e : map.entrySet()) {
            if (maxEntry.getValue().equals(e.getValue())) {
                s.add("(" + e.getKey() + "," + e.getValue() + ")");
            }
        }
        Collections.sort(s);
        return s;
    }

}