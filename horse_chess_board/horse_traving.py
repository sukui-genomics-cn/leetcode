chess_list = []  # 存放落子位置的公共列表
times = 0  # 记录递归函数被调用次数的公共变量
 
 
def return_location(x: int, y: int):
    """
    返回可移动位置有那些
    :param x: x坐标
    :param y: y坐标
    :return: 可移动位置列表
    """
    l_list = []  # 创建一个列表来存储可移动位置
    if x - 2 > 0 and y + 1 < 9:  # 判断此位置是否超出棋盘
        l_list.append((x - 2, y + 1))  # 没有超出棋盘，则放入列表中
    if x - 1 > 0 and y + 2 < 9:
        l_list.append((x - 1, y + 2))
    if x + 1 < 9 and y + 2 < 9:
        l_list.append((x + 1, y + 2))
    if x + 2 < 9 and y + 1 < 9:
        l_list.append((x + 2, y + 1))
    if x + 2 < 9 and y - 1 > 0:
        l_list.append((x + 2, y - 1))
    if x + 1 < 9 and y - 2 > 0:
        l_list.append((x + 1, y - 2))
    if x - 1 > 0 and y - 2 > 0:
        l_list.append((x - 1, y - 2))
    if x - 2 > 0 and y - 1 > 0:
        l_list.append((x - 2, y - 1))
    return l_list  # 返回可移动位置列表
 
 
def full_chess_board(x: int, y: int):
    """
    使用递归找出能填满整个棋盘的线路
    :param x: x坐标
    :param y: y坐标
    :return: bool，当递归深度为64时返回True，否则返回False
    """
    global times  # 声明公共变量
    times += 1  # 每调用一次递归函数times加一
    chess_list.append((x, y))  # 把落子位置放入公共列表中
    if len(chess_list) == 64:  # 判断棋盘所有位置是否都已落子（递归深度是否为64）
        return True  # 返回True
    for i in return_location(x, y):  # 使用循环把可移动位置逐个输出
        if i not in chess_list:  # 判断该位置在列表chess_list中是否存在
            if full_chess_board(i[0], i[1]):  # 使用递归判断向下一个可移动位置移动是否可行（不断递归下去，能否达到64的深度）
                return True  # 返回True
    chess_list.remove((x, y))  # 此位置所有的分支都为绝路时，把此位置从公共列表中移除
    return False  # 返回False
 
 
def print_chess_order(x: int, y: int):
    """
    在棋盘中打印落子顺序
    :param x: x坐标
    :param y: y坐标
    :return:
    """
    if type(x) != int or type(y) != int:  # 判断x和y的类型是不是int
        raise TypeError('x, y type must int')  # 不是int则抛出类型错误
    if x < 1 or x > 8 or y < 1 or y > 8:  # 判断x和y的值是否在[1, 8]范围内
        raise ValueError('x, y range [1, 8]')  # 超出范围抛出值错误
    full_chess_board(x, y)  # 执行递归查找填充路线
    print(chess_list)  # 打印出棋子行走坐标
    print(times)  # 打印出使用递归函数的次数
    row_list = [[0] * 9 for i in range(9)]  # 创建一个二维列表来存放行走顺序（1~64）
    number = 1  # 第一个落子位置序号为1
    for i in chess_list:  # 通过循环来逐个输出落子位置
        row_list[i[0]][i[1]] = number  # 在棋盘对应位置记录落子序号
        number += 1  # 循环一次序号加一
    print('-' * 41)  # 打印棋盘上边缘线
    for i in range(1, 9):  # 通过循环逐个输出1到8的数字
        for j in range(1, 9):  # 通过循环逐个输出1到8的数字
            print(f'|{row_list[i][j]:^4}', end='')  # 取出二维列表中棋盘对应位置的序号，并打印出来
        print('|')  # 棋盘右边界线
        print('-' * 41)  # 打印棋盘底部边界线
 
 
print_chess_order(2, 2)