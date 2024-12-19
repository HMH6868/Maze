import pygame
import random
import time
from collections import deque
import tkinter as tk
from tkinter import simpledialog
import heapq

# Hàm hỏi người dùng nhập kích thước mê cung
def get_maze_size():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    rows = simpledialog.askinteger("Maze Size", "Nhập số hàng của mê cung (ROWS):", minvalue=10, maxvalue=100)
    cols = simpledialog.askinteger("Maze Size", "Nhập số cột của mê cung (COLS):", minvalue=10, maxvalue=100)
    root.destroy()
    return rows, cols

# Nhận kích thước mê cung từ người dùng
ROWS, COLS = get_maze_size()

# Kích thước cửa sổ và ô
CELL_SIZE = 800 // max(ROWS, COLS)  # Tự động điều chỉnh kích thước ô dựa trên kích thước mê cung
SCREEN_WIDTH = CELL_SIZE * COLS
SCREEN_HEIGHT = CELL_SIZE * ROWS

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
DFS_COLOR = (200, 200, 255)
BFS_COLOR = (255, 200, 200)

# Mảng mê cung, 0 là đường đi, 1 là tường
maze = [[1 for _ in range(COLS)] for _ in range(ROWS)]

# Cổng vào và cổng ra
start = (0, 0)
end = (ROWS - 1, COLS - 1)

# Khởi tạo pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Maze Solver with DFS and BFS")

# Hàm vẽ mê cung với đường đi
def draw_maze(path=None):
    for y in range(ROWS):
        for x in range(COLS):
            color = WHITE if maze[y][x] == 0 else BLACK
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, GREEN, (start[1] * CELL_SIZE, start[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (end[1] * CELL_SIZE, end[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Vẽ đường đi nếu có
    if path:
        for step in path:
            pygame.draw.rect(screen, BLUE, (step[1] * CELL_SIZE, step[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

# Hàm tạo mê cung khả thi
def generate_maze():
    global maze, start, end
    maze = [[1 for _ in range(COLS)] for _ in range(ROWS)]  # Khởi tạo toàn bộ là tường

    # Đảm bảo cổng vào và cổng ra
    start = (0, 0)
    end = (ROWS - 1, COLS - 1)

    # Sử dụng DFS để tạo một đường đi
    def dfs(x, y):
        maze[x][y] = 0  # Đánh dấu là đường đi
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Các hướng: phải, xuống, trái, lên
        random.shuffle(directions)  # Xáo trộn hướng để đảm bảo ngẫu nhiên

        for dx, dy in directions:
            nx, ny = x + dx * 2, y + dy * 2  # Nhảy 2 ô để tạo đường đi
            if 0 <= nx < ROWS and 0 <= ny < COLS and maze[nx][ny] == 1:  # Nếu là tường
                maze[nx][ny] = 0  # Đánh dấu ô đó là đường đi
                maze[x + dx][y + dy] = 0  # Xóa tường giữa 2 ô
                dfs(nx, ny)  # Tiếp tục từ ô tiếp theo

    # Khởi tạo đường đi từ start
    dfs(start[0], start[1])

    # Đảm bảo điểm cuối (end) được kết nối
    if maze[end[0]][end[1]] == 1:  # Nếu end là tường, phá tường lân cận để tạo kết nối
        maze[end[0]][end[1]] = 0
        if end[0] > 0 and maze[end[0] - 1][end[1]] == 1:  # Nếu phía trên là tường
            maze[end[0] - 1][end[1]] = 0
        elif end[1] > 0 and maze[end[0]][end[1] - 1] == 1:  # Nếu phía trái là tường
            maze[end[0]][end[1] - 1] = 0

def solve_maze_with_arrows(path, algorithm):
    if path is None:
        print(f"{algorithm}: Không có đường đi!")
        return

    print(f"{algorithm}: Đường đi tìm được có độ dài: {len(path)}")

    for i in range(len(path) - 1):
        current = path[i]
        next = path[i + 1]

        # Tính toán vị trí mũi tên
        x1, y1 = current[1] * CELL_SIZE + CELL_SIZE // 2, current[0] * CELL_SIZE + CELL_SIZE // 2
        x2, y2 = next[1] * CELL_SIZE + CELL_SIZE // 2, next[0] * CELL_SIZE + CELL_SIZE // 2

        # Vẽ đường và mũi tên chỉ hướng
        pygame.draw.line(screen, BLUE, (x1, y1), (x2, y2), 3)  # Đường nối giữa 2 ô
        draw_arrow((x1, y1), (x2, y2))  # Gọi hàm vẽ mũi tên
        pygame.display.update()
        time.sleep(0.05)

    # Đánh dấu ô cuối cùng
    last = path[-1]
    pygame.draw.rect(screen, RED, (last[1] * CELL_SIZE + CELL_SIZE // 4, last[0] * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))
    pygame.display.update()


def draw_arrow(start, end):
    """Vẽ mũi tên từ start đến end."""
    x1, y1 = start
    x2, y2 = end

    # Tính góc và vị trí của đầu mũi tên
    dx, dy = x2 - x1, y2 - y1
    length = (dx**2 + dy**2)**0.5
    unit_dx, unit_dy = dx / length, dy / length

    # Đầu mũi tên
    arrow_head_x = x2 - unit_dx * 10
    arrow_head_y = y2 - unit_dy * 10

    # Góc vuông để tạo cạnh mũi tên
    perp_dx, perp_dy = -unit_dy, unit_dx

    # Các điểm tạo thành mũi tên
    arrow_points = [
        (x2, y2),  # Đỉnh mũi tên
        (arrow_head_x + perp_dx * 5, arrow_head_y + perp_dy * 5),  # Cạnh trái
        (arrow_head_x - perp_dx * 5, arrow_head_y - perp_dy * 5),  # Cạnh phải
    ]

    # Vẽ mũi tên
    pygame.draw.polygon(screen, BLUE, arrow_points)



# Thuật toán DFS tìm đường và hiển thị
def dfs_solver_with_visualization(start, end):
    stack = [start]
    visited = set()
    parent = {start: None}

    while stack:
        current = stack.pop()

        # Hiển thị ô đang duyệt
        if current not in visited:
            pygame.draw.rect(screen, DFS_COLOR, (current[1] * CELL_SIZE, current[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.display.update()
            time.sleep(0.01)

        if current == end:
            break
        visited.add(current)

        # Xáo trộn các hướng duyệt để DFS tạo ra các đường đi khác nhau
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(neighbors)

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and maze[nx][ny] == 0 and (nx, ny) not in visited:
                stack.append((nx, ny))
                parent[(nx, ny)] = current

    # Truy vết lại đường đi
    if end not in parent:
        return None  # Không tìm thấy đường đi

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path

# Thuật toán BFS tìm đường và hiển thị
def bfs_solver_with_visualization(start, end):
    queue = deque([start])
    visited = set()
    parent = {start: None}

    while queue:
        current = queue.popleft()

        # Hiển thị ô đang duyệt
        if current not in visited:
            pygame.draw.rect(screen, BFS_COLOR, (current[1] * CELL_SIZE, current[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.display.update()
            time.sleep(0.01)

        if current == end:
            break
        visited.add(current)

        # Các hướng di chuyển
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and maze[nx][ny] == 0 and (nx, ny) not in visited:
                queue.append((nx, ny))
                parent[(nx, ny)] = current

    # Truy vết lại đường đi
    if end not in parent:
        return None  # Không tìm thấy đường đi

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path



# Hàm giải mê cung bằng thuật toán Dijkstra
def dijkstra_solver_with_visualization(start, end):
    # Khởi tạo bảng khoảng cách vô cùng lớn cho tất cả các ô
    distance = { (r, c): float('inf') for r in range(ROWS) for c in range(COLS) }
    distance[start] = 0  # Đặt khoảng cách bắt đầu bằng 0
    parent = {start: None}  # Lưu trữ các ô trước đó để tái hiện đường đi
    visited = set()  # Tập các ô đã duyệt
    pq = [(0, start)]  # Hàng đợi ưu tiên (giá trị khoảng cách, tọa độ)

    while pq:
        current_distance, current = heapq.heappop(pq)  # Lấy ô có khoảng cách nhỏ nhất
        
        if current == end:  # Nếu tìm được đích
            break
        
        # Duyệt các hướng (phải, trái, xuống, lên)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and maze[nx][ny] == 0 and (nx, ny) not in visited:
                new_distance = current_distance + 1  # Khoảng cách từ start tới ô này
                if new_distance < distance[(nx, ny)]:
                    distance[(nx, ny)] = new_distance
                    parent[(nx, ny)] = current
                    heapq.heappush(pq, (new_distance, (nx, ny)))

        # Hiển thị ô đang duyệt
        if current not in visited:
            pygame.draw.rect(screen, DFS_COLOR, (current[1] * CELL_SIZE, current[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.display.update()
            time.sleep(0.01)

        visited.add(current)

    # Truy vết lại đường đi từ end về start
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path



# Hàm vẽ đường đi từng bước và giữ lại
def solve_maze(path, algorithm):
    if path is None:
        print(f"{algorithm}: Không có đường đi!")
        return

    print(f"{algorithm}: Đường đi tìm được có độ dài: {len(path)}")
    for step in path:
        pygame.draw.rect(screen, BLUE, (step[1] * CELL_SIZE, step[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.update()
        time.sleep(0.05)



def main():
    global maze, start, end
    running = True
    clock = pygame.time.Clock()
    path = []
    algorithm = None  # Lưu thuật toán được chọn: "DFS", "BFS" hoặc "Dijkstra"

    while running:
        screen.fill(BLACK)
        draw_maze(path)  # Vẽ mê cung cùng với đường đi

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Nhấn R để tạo mê cung mới
                    generate_maze()
                    path = []  # Xóa đường đi khi tạo mê cung mới
                if event.key == pygame.K_d:  # Nhấn D để chọn DFS
                    algorithm = "DFS"
                if event.key == pygame.K_b:  # Nhấn B để chọn BFS
                    algorithm = "BFS"
                if event.key == pygame.K_i:  # Nhấn I để chọn Dijkstra
                    algorithm = "Dijkstra"
                if event.key == pygame.K_s:  # Nhấn S để bắt đầu tìm đường
                    if algorithm == "DFS":
                        path = dfs_solver_with_visualization(start, end)
                        solve_maze_with_arrows(path, "DFS")
                    elif algorithm == "BFS":
                        path = bfs_solver_with_visualization(start, end)
                        solve_maze_with_arrows(path, "BFS")
                    elif algorithm == "Dijkstra":
                        path = dijkstra_solver_with_visualization(start, end)
                        solve_maze_with_arrows(path, "Dijkstra")
                    else:
                        print("Hãy chọn thuật toán bằng cách nhấn D (DFS), B (BFS) hoặc I (Dijkstra)!")  # Cập nhật hướng dẫn
                if event.key == pygame.K_x:  # Nhấn X để xóa đường đi
                    path = []  # Xóa đường đi hiện tại

        pygame.display.update()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    generate_maze()
    main()
