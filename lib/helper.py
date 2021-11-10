def grid_to_str(grid):
    s = ""
    for row in grid:
        for sq in row:
            s += " " + str(sq)
        s += "\n"
    return s


def copy_2d(grid):
    copied = list()
    for row in grid:
        copied.append(list(row))
    return copied


def text_list_flatten(text_list):
    text = ""
    for s in text_list:
        if not isinstance(s, str):
            print("")
        text += s + ", "
    return text
