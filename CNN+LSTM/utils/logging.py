import logging

# 这段代码定义了一个函数 format_logger，用于格式化日志记录器的输出格式。它自定义了日志输出的格式，并应用到传入的日志记录器上。
# \033[31m 和 \033[0m 是终端颜色控制符，表示输出红色字体的文本
# %(asctime)s: 时间戳（记录日志的时间）
# %(levelname)s: 日志级别（例如 INFO, WARNING, ERROR）
# %(message)s: 日志的主要信息部分

# def format_logger(logger, fmt="\033[31m[%(asctime)s %(levelname)s]\033[0m%(message)s"):

def output_logger_to_file(logger, output_path, fmt="[%(asctime)s %(levelname)s]%(message)s"):
    handler = logging.FileHandler(output_path, mode='w', encoding="UTF-8")  # 这里创建了一个文件处理器 (FileHandler)，它负责将日志输出到指定的文件 output_path 中，并使用 UTF-8 编码。
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)  # 将文件处理器添加到日志记录器中。日志记录器本身可以有多个处理器（如控制台输出、文件输出等），每个处理器可以将日志输出到不同的地方。