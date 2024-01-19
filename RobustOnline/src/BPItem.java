/**
 * Represents an item used in the bin packing problem having a size
 * 
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class BPItem{
   
    /**
     * The name of the item
     */
    String id;
    
    /** 
     * The size of the item
     */
    double size;
    
    /**
     * How often this item exists. 
     * 
     * This is simply used to prevent the need of copying a lot of
     * items.
     */
    double multiplicity;

    /**
     * Creates a new item
     * 
     * @param id the name of the new item
     * @param size the size of the new item
     * @param multiplicity the multiplicity of the new item
     */
    public BPItem(String id, double size, double multiplicity){
        this.id   = id;
        this.size = size;
        this.multiplicity = multiplicity;
    }
    
    /**
     * Test whether this equals another object
     * 
     * @param o the object to compare with
     * @return <code>true</code> if o is an <code>BPItem</code> and both
     *         items have the same name;
     *         <code>false</code> otherwise.
     */
    public boolean equals(Object o){
        return (o instanceof BPItem && id.equals(((BPItem) o).id));
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
