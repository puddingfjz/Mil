
def get_best_solution(filename):
    try:
        with open(filename, 'r', errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if 'Best group seq: ' in line:
                    pos = line.find(']]') + len(']]')
                    return line[:pos]
    except Exception as e:
        return None


def check_consistency(file_names1, file_names2):
    for name1, name2 in zip(file_names1, file_names2):
        sol1 = get_best_solution(name1)
        sol2 = get_best_solution(name2)

        print(f"name1: {name1}")
        print(f"sol1: {sol1}")
        print(f"sol2: {sol2}\n")

        if sol1 != sol2:
            print(f"name1: {name1}")
            print(f"sol1: {sol1}")
            print(f"sol2: {sol2}\n")


