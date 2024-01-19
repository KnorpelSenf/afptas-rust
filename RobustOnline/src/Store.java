/**
 * This represents a triple containing two numbers and a reference to a
 * tree-like structure of those triples.
 *
 * This is used to store information about profits and sizes in the
 * knapsack problem.
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class Store{
    /**
     * The sum of the profits
     */
    double p;
    
    /**
     * The sum of the sizes
     */
    double a;
    
    /**
     * The {@link StoreTree} containing the previous items
     */
    StoreTree tree;
    
    /**
     * Creates a new Store with no predecessors
     *
     * @param p the sum of profits
     * @param a the sum of sizes
     */
    public Store(double p, double a){
        this.p = p;
        this.a = a;
        tree = null;
    }
    
    /**
     * Creates a new Store with an predecessor
     *
     * @param o the predecessor of the new Store
     * @param pj the profit of the added item
     * @param aj the size of the added item
     * @param j the added item
     */
    public Store(Store o, double pj, double aj, KPItem j){
        this.p = o.p+pj;
        this.a = o.a+aj;
        tree = new StoreTree(j,o.tree);
    }
}
