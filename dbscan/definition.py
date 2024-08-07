class Distance:
    def __init__(self):
        return

    def euclidean(p1, p2):
        if len(p1) != 2:
            raise ValueError("wrong p1; Euclidean require two value")
        if len(p2) != 2:
            raise ValueError("wrong p2; Euclidean require two value")
        squared1 = p1 ** 2
        squared2 = p2 ** 2
        return (squared1 + squared2) ** 0.5
         
