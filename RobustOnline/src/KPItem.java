/**
 * Represents an item used in the knapsack problem having a size and a profit
 * 
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class KPItem implements Comparable<KPItem>{
 
    /**
     * The name of the item
     */
    String id;
    
    /**
     * The profit of the item
     */
    double profit;
    
    /**
     * The size of the item
     */
    double size;
    
    /**
     * How often this item exists. 
     *
     * This is used when producing copied of the item
     */
    int multiplicity;
    
    /**
     * Creates a new item
     * 
     * @param id the name of the new item
     * @param profit the profit of the new item
     * @param size the size of the new item
     */
    public KPItem(String id, double profit, double size){
        this.id     = id;
        this.profit = profit;
        this.size   = size;
        this.multiplicity = 1;
    }

    /**
     * Creates a new item by copying another one
     *
     * @param o the <code>KPItem</code> to be copied
     * @param multiplicity the multiplicity of the new item
     */
    public KPItem(KPItem o, int multiplicity){
        this.id     = o.id;
        this.profit = o.profit;
        this.size   = o.size;
        this.multiplicity = multiplicity;
    }

    /**
     * Compares the profit/size ratio of this item to the ration of
     * another <code>KPItem</code>
     *
     * @param other the KPItem to compare with 
     * @return the value <code>0</code> if both ratios are equal; 
     *         a value less than <code>0</code> if the other item has a
     *         smaller ratio;
     *         a value greater than <code>0</code> if the other item 
     *         has a bigger ratio
     */
    public int compareTo(KPItem other){ 
        double v1 = profit/size; 
        double v2 = other.profit/other.size;
        return Double.compare(v2,v1); }
    
    /**
     * Tests whether this equals another object
     *
     * @param o the object to compare with
     * @return <code>true</code> if o is an <code>KPItem</code> and both
     *         items have the same name;
     *         <code>false</code> otherwise.
     */
   
    public boolean equals(Object o){
        return (o instanceof KPItem && id.equals(((KPItem) o).id));
    }

    /**
     * Computes the hash code of the item
     *
     * @return the hash code which is given by the hash code of the name
     */
    public int hashCode(){
        return id.hashCode();
    }
}







