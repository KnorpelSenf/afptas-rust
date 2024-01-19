This is an implementation of the static online algorithm bin packing with migration
described in "[Fully Dynamic Bin Packing Revisited](https://arxiv.org/pdf/1411.0960.pdf)" by Berndt, Jansen, and Klein (APPROX 2015; Math. Program. 2018).

For compilation, the [Apache Commons Math library](http://commons.apache.org/proper/commons-math/) needs to be downloaded.
To compile, simply use
```shell
javac -cp commons-math3.jar:. TestBinPacking
```

To use the program, simply use:
```shell
java -c commons-math3.jar:. TestBinPacking inputFile numberOfItems epsilon
```

The inputFile is an XML file with the structure
```xml
<problem>
<number size="<SIZE>"/>
<number size="<SIZE>"/>
<number size="<SIZE>"/>
<number size="<SIZE>"/>
...
<number size="<SIZE>"/>
<list/>
</problem>
```


For example, an instance containing three items with size 0.5, 0.4, and 0.2 is given as
```xml
<problem>
<number size="0.5"/>
<number size="0.4"/>
<number size="0.2"/>
<list/>
</problem>
```

The output is given as CSV file, where the first column corresponds to the time, the second column gives the total migration at this time, the third column gives the processing time needed only for this time step, and the fourth column gives the number of bins currently in use. 

For example, the output of the example above with epsilon=0.05 is
```shell
counter migrationFactor timeApprox valueApprox
2 0.5 5.0 1.0
3 0.9 25.0 2.0
Finished NEW
counter migrationFactor timeApprox valueApprox
2 0.5 0.0 1.0
3 0.9 0.0 2.0
Finished OLD
```
Here, the first part is the result of the algorithm with the rounding proposed in the paper of Berndt et al. and the second part is the result of the algorithm with the rounding given in "A Robust APTAS for the Classical Bin Packing Problem(http://math.haifa.ac.il/lea/ICALP_APTAS.pdf)" by Epstein and Levin (ICALP 2006; Math. Program. 2009). 
