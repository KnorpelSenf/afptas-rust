/**
 * Uses an array to give input to the online bin packing problem
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
class TestInput implements Input{

    /**
     * Stores the unique id
     */
    int counter = 0;

    /**
     * When to stop generating
     */
    int stop;
    
    /**
     * The different sizes to use
     */
    double[] sizes;

    /**
     * Creates a new TestInput instance
     *
     * @param stop when to stop giving items
     * @param sizes the sizes of the problem stored in an array
     */
    public TestInput(int stop, double[] sizes){
        counter = 0;
        this.stop = stop;
        this.sizes = sizes;
    }

    /**
     * Returns the next item with a given size
     *
     * @return a random generated BPItem
     */
    public BPItem getNextItem(){
        if (counter > stop){
            return null;
        }
        double size = sizes[counter];
        String id = ""+counter;
        double multiplicity = 1.0;
        counter++;
        return new BPItem(id,size,multiplicity);
    }
}





