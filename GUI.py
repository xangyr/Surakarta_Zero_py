import tkinter as tk


root = tk.Tk()
person_flag = 1
piece_color="black"
PIECE_SIZE=10

click_x = -1
click_y = -1

root.title("Surakarta")
root.geometry("760x550")

pieces_x = [i for i in range(120, 480, 60)]
pieces_y = [i for i in range(120, 480, 60)]

to_x=-1
to_y=-1

coor_black = []
coor_white = []








def showChange(color):
    global piece_color
    piece_color = color
    side_canvas.delete("omrom")
    side_canvas.create_oval(110 - PIECE_SIZE, 25 - PIECE_SIZE,
                        110 + PIECE_SIZE, 25 + PIECE_SIZE,
                        fill = piece_color, tags = ("show_piece"))

def coorFrom(event):  #return coordinates of cursor 返回光标坐标
    global click_x, click_y
    click_x = event.x
    click_y = event.y


def coorTo(event):
    global to_x, to_y
    to_x = event.x
    to_y = event.y
    coorJudge()

def coorJudge():
    global click_x, click_y, to_x, to_y
    coor = coor_black + coor_white
    global person_flag, show_piece
    #print("x = %s, y = %s" % (click_x, click_y))
    item_from = canvas.find_closest(click_x, click_y)
    item_to = canvas.find_closest(to_x,to_y)
    tags_from_tuple = canvas.gettags(item_from)
    tags_to_tuple = canvas.gettags(item_to)
    
    if len(tags_from_tuple) > 1 and len(tags_to_tuple)>1:
        tags_from_list = list(tags_from_tuple)
        tags_to_list=list(tags_to_tuple)
        coor_from_list = tags_from_list[:2]
        coor_to_list=tags_to_list[:2]
        try:
            for i in range(len(coor_from_list)):
                coor_from_list[i] = int(coor_from_list[i])
                coor_to_list[i] = int(coor_to_list[i])
        except ValueError:
            pass
        else:
            coor_from_tuple = tuple(coor_from_list)
            coor_to_tuple = tuple(coor_to_list)
            (click_x, click_y) = coor_from_tuple
            (to_x, to_y) = coor_to_tuple
            #print("tags = ", tags_tuple, "coors = ", coor_tuple)
            if ( to_x in pieces_x )and( to_y in pieces_y )and( click_x in pieces_x )and( click_y in pieces_y ) and not(to_x==click_x and to_y==click_y):   # from xy and to xy are in valid point set and not equal
                #print("True")
                if person_flag != 0:
                    if person_flag == 1 and ((click_x,click_y) in coor_black) and ((to_x,to_y) not in coor_black) and check_valid():
                        print('black')
                        movePiece("black")
                        showChange("white")
                        var.set("move white stone")
                        person_flag *= -1
                    elif person_flag == -1 and ((click_x,click_y) in coor_white) and ((to_x,to_y) not in coor_white) and check_valid():
                        print('white')
                        movePiece("white")
                        showChange("black")
                        var.set("move black stone")
                        person_flag *= -1
            else:
                print("False")



def check_valid(): #检查from x,y to x,y是否符合苏棋标准
    return True

def movePiece(piece_color):
    global coor_black, coor_white
    
    if (to_x,to_y) in coor_white:
        
        item_from = canvas.find_closest(click_x,click_y)
        canvas.itemconfig(item_from,fill = '')
        item_to = canvas.find_closest(to_x,to_y)
        canvas.itemconfig(item_to,fill = piece_color)
        coor_white.remove((to_x,to_y))
        coor_black.remove((click_x,click_y))
        coor_black.append((to_x,to_y))
        
    elif (to_x,to_y) in coor_black:
        
        item_from = canvas.find_closest(click_x,click_y)
        canvas.itemconfig(item_from,fill = '')
        item_to = canvas.find_closest(to_x,to_y)
        canvas.itemconfig(item_to,fill = piece_color)
        coor_black.remove((to_x,to_y))
        coor_white.remove((click_x,click_y))
        coor_white.append((to_x,to_y))
    else:
        
        item_from = canvas.find_closest(click_x,click_y)
        canvas.itemconfig(item_from,fill = '')
        item_to = canvas.find_closest(to_x,to_y)
        canvas.itemconfig(item_to,fill = piece_color)
        if (click_x,click_y) in coor_white:
            coor_white.remove((click_x,click_y))
            coor_white.append((to_x, to_y))
        else:
            coor_black.remove((click_x,click_y))
            coor_black.append((to_x, to_y))
    detect_end()
        

    
def gReset(event):
    global person_flag, coor_black, coor_white, piece_color
    person_flag = 1       #打开落子开关
    var.set("black stone move")      #还原提示标签
    var1.set("")          #还原输赢提示标签
    var2.set("")          #还原游戏结束提示标签
    showChange("black")   #还原棋子提示图片
    
    coor_black = []       #清空黑棋坐标存储器
    coor_white = []       #清空白棋坐标存储器
    for i in pieces_x:
        for j in pieces_y:
            if j==120 or j==180: 
                item = canvas.find_closest(i,j)
                canvas.itemconfig(item,fill = 'black')
                coor_black.append((i,j))
            
            elif j==240 or j==300:
                item = canvas.find_closest(i,j)
                canvas.itemconfig(item,fill = '')
            elif j==360 or j==420:
                item = canvas.find_closest(i,j)
                canvas.itemconfig(item,fill = 'white')
                coor_white.append((i,j))


def detect_end():
    if len(coor_white)==0 or len(coor_black)==0:
        person_flag=0
        pushMessage()
        return True
    return False


def pushMessage():
    if person_flag == -1:
        var1.set('White stone win')
    elif person_flag == 1:
        var1.set('Black stone win')
    var2.set('Game Over')






"""棋子提示标签"""
side_canvas = tk.Canvas(root, width = 220, height = 50)
side_canvas.grid(row = 0, column = 1)
side_canvas.create_oval(110 - PIECE_SIZE, 25 - PIECE_SIZE,
                        110 + PIECE_SIZE, 25 + PIECE_SIZE,
                        fill = piece_color, tags = ("show_piece") )


"""棋子提示标签"""
var = tk.StringVar()
var.set("black stone round")
person_label = tk.Label(root, textvariable = var, width = 12, anchor = tk.CENTER, 
                        font = ("Arial", 10) )
person_label.grid(row = 1, column = 1)


"""输赢提示标签"""
var1 = tk.StringVar()
var1.set("")
result_label = tk.Label(root, textvariable = var1, width = 12, height = 4, 
                        anchor = tk.CENTER, fg = "red", font = ("Arial", 25) )
result_label.grid(row = 2, column = 1, rowspan = 2)

"""游戏结束提示标签"""
var2 = tk.StringVar()
var2.set("")
game_label = tk.Label(root, textvariable = var2, width = 12, height = 4, 
                        anchor = tk.CENTER, font = ("Arial", 18) )
game_label.grid(row = 4, column = 1)

"""重置按钮"""
reset_button = tk.Button(root, text = "replay", font = 5, 
                          width = 8)####command=reset
reset_button.grid(row = 5, column = 1)
reset_button.bind("<Button-1>", gReset)

"""棋盘绘制"""
canvas = tk.Canvas(root, bg = "grey", width = 540, height = 540)
canvas.bind("<Button-1>", coorFrom )  
canvas.bind("<Button-3>", coorTo )
canvas.grid(row = 0, column = 0, rowspan = 6)


for i in range(6):
    canvas.create_line(120, (60 * i + 120), 420, (60 * i + 120))
    canvas.create_line((60 * i + 120), 120, (60 * i + 120), 420)

canvas.create_arc(0,0,240,240,start=0, extent=270)
canvas.create_arc(60,60,180,180,start=0, extent=270)
canvas.create_arc(0,300,240,540,start=90, extent=270)
canvas.create_arc(60,360,180,480,start=90, extent=270)
canvas.create_arc(300,0,540,240,start=180, extent=-270)
canvas.create_arc(360,60,480,180,start=180, extent=-270)
canvas.create_arc(300,300,540,540,start=90, extent=-270)
canvas.create_arc(360,360,480,480,start=90, extent=-270)


for i in pieces_x:
    for j in pieces_y:
        if j==120 or j==180: 
            canvas.create_oval(i - PIECE_SIZE, j - PIECE_SIZE,
                               i + PIECE_SIZE, j + PIECE_SIZE,
                               width = 0,fill='black', tags = (str(i), str(j)))
            coor_black.append((i,j))
            
        elif j==240 or j==300:
            canvas.create_oval(i - PIECE_SIZE, j - PIECE_SIZE,
                               i + PIECE_SIZE, j + PIECE_SIZE,
                               width = 0,fill='', tags = (str(i), str(j)))

        elif j==360 or j==420:
            canvas.create_oval(i - PIECE_SIZE, j - PIECE_SIZE,
                               i + PIECE_SIZE, j + PIECE_SIZE,
                               width = 0,fill='white', tags = (str(i), str(j)))
            coor_white.append((i,j))

##item_from = canvas.find_closest(120,120)
##canvas.itemconfig(item_from,fill = 'white')
##检查用


