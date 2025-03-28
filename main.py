"""
主程序入口，用于训练或游玩
"""
import argparse
import os

def main():
    """
    主程序入口
    """
    parser = argparse.ArgumentParser(description="Super Mario Bros Reinforcement Learning")
    parser.add_argument("action", type=str, choices=["train", "play"], help="Action to perform (train or play)")
    
    # 如果没有提供任何参数，显示帮助信息
    if os.sys.argv[1:] == []:
        parser.print_help()
        return
    
    args, remaining = parser.parse_known_args()
    
    if args.action == "train":
        # 调用训练脚本
        import train
        # 将剩余参数传递给训练脚本
        # sys.argv[0] 是脚本名称，后面跟着各种参数
        os.sys.argv = [train.__file__] + remaining
        train.main()
    elif args.action == "play":
        # 调用游玩脚本
        import play
        # 将剩余参数传递给游玩脚本
        os.sys.argv = [play.__file__] + remaining
        play.main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
