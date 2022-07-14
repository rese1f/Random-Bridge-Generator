import bpy
cp = bpy.data.texts["components.py"].as_module()

if __name__ == '__main__':
    # Random
    ### YOUR CODE HERE

    ### CODE END

    # Initialize structures
    superStructure = cp.SuperStructure()
    subStructure = cp.SubStructure()
    deck = cp.Deck()

    # Generate Objects
    superStructure.randomGenerate()
    subStructure.randomGenerate()
    deck.randomGenerate()