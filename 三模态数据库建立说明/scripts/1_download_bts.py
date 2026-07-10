import os
import time
import subprocess

# 配置（基于脚本位置，与 process_bts_zips.py 的 ZIP_DIR 对齐）
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "bts_2016_2025")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 代理配置（BTS 服务器在美国政府网络，国内直连会卡死）
PROXY = "http://127.0.0.1:7892"


def download_month(year, month, max_retry=5):
    filename = f"{year}_{month:02d}.zip"
    save_path = os.path.join(OUTPUT_DIR, filename)

    # 已下载就跳过（断点续传，>1KB 视为完整）
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
        return True

    url = (
        f"https://transtats.bts.gov/PREZIP/"
        f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    )

    for retry in range(max_retry):
        try:
            subprocess.run(
                [
                    "curl.exe", "-x", PROXY, "-k", "-L",
                    "--connect-timeout", "30",
                    "--max-time", "600",
                    "-#", "-o", save_path, url,
                ],
                check=True,
            )
            sz = os.path.getsize(save_path) / 1e6
            print(f" OK {year}-{month:02d} ({sz:.1f}MB)")
            return True
        except subprocess.CalledProcessError:
            # 删除不完整的半成品
            if os.path.exists(save_path):
                os.remove(save_path)
            wait = 8 * (retry + 1)
            print(f"  重试 {retry+1}/{max_retry} {year}-{month:02d}, {wait}s")
            time.sleep(wait)

    print(f" 失败 {year}-{month:02d}")
    return False


if __name__ == "__main__":
    print(" 开始下载 BTS 2016–2025 全量（curl + 代理）")

    for year in range(2016, 2026):
        print(f"\n===== {year}年 =====")
        for month in range(1, 13):
            download_month(year, month)

    print("\n 全部完成！")
