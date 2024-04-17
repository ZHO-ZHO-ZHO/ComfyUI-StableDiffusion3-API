
![ComfyUI_temp_xcgvh_00014_](https://github.com/ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API/assets/140084057/39a8e52b-0df3-462a-b2e5-2bcae481f8ea)


# ComfyUI Stable Diffusion 3 API

<!---
![Dingtalk_20240418023810](https://github.com/ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API/assets/140084057/071031e5-0a53-4257-adf1-ed61e80fa9f6)
--->

![Dingtalk_20240418031631](https://github.com/ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API/assets/140084057/3c3f9b8a-c219-47c4-88d3-b52cece676da)

![Dingtalk_20240418031914](https://github.com/ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API/assets/140084057/8b85f15c-b3f6-421a-831e-e05fccc097a6)


## 项目介绍 | Info

- 通过 API 将 Stable Diffusion 3 引入 ComfyUI

- Stable Diffusion 3：目前通过 API 开放，[详情](https://stability.ai/news/stable-diffusion-3-api?utm_source=twitter&utm_medium=website&utm_campaign=blog)，API申请：[Stability API key](https://platform.stability.ai/account/keys)，每个账户提供 25 个免费积分

- 模型：
   - SD3 6.5 积分/张
   - SD3 Turbo 4 积分/张 
  
- 版本：V1.0 支持多种比例的 SD3 文生图（SD3 Turbo 和 图生图功能还有问题）


![Dingtalk_20240418004328](https://github.com/ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API/assets/140084057/89952a6b-b46b-419b-a429-2771621998a7)


## 节点说明 | Features

- 🔥Stable Diffusion 3
    - positive：正向提示词
    - negative：负向提示词
    - aspect_ratio：画面比例，共 9 种："21:9", "16:9", "5:4", "3:2", "1:1", "2:3", "4:5", "9:16", "9:21"
    - mode：文生图 或 图生图（还未支持）
    - model：SD3 或 SD3 Turbo（还未支持）
    - seed：种子
    - image（非必要，用于图生图，还未支持）
    - strength（非必要，用于图生图，还未支持）
    
![Dingtalk_20240418025505](https://github.com/ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API/assets/140084057/47b122f9-3864-47fe-ae34-f2a25275a701)


## 安装 | Install

- 使用前请先申请 API ：[Stability API key](https://platform.stability.ai/account/keys)，每个账户提供 25 个免费积分

- 将 Stability API key 添加到 config.json 文件中，运行时会自动加载

- 推荐使用管理器 ComfyUI Manager 安装（On The Way）

- 手动安装：
    1. `cd custom_nodes`
    2. `git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API`
    3. 重启 ComfyUI


## 工作流 | Workflows

V1.0

  - [V1.0 SD3 API]()

    ![Dingtalk_20240418030657](https://github.com/ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API/assets/140084057/c8c1f265-7f59-430d-ac00-0713019cbe44)
    <!---
    ![Dingtalk_20240418025716](https://github.com/ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API/assets/140084057/833519ff-9480-4802-a43b-fa9a835fa7ef)
    --->



## 更新日志

- 202418

  V1.0 支持多种比例的 SD3 文生图（SD3 Turbo 和 图生图功能还有问题） 

  创建项目
  


## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API&type=Date)](https://star-history.com/#ZHO-ZHO-ZHO/ComfyUI-StableDiffusion3-API&Date)


## 关于我 | About me

📬 **联系我**：
- 邮箱：zhozho3965@gmail.com
- QQ 群：839821928

🔗 **社交媒体**：
- 个人页：[-Zho-](https://jike.city/zho)
- Bilibili：[我的B站主页](https://space.bilibili.com/484366804)
- X（Twitter）：[我的Twitter](https://twitter.com/ZHOZHO672070)
- 小红书：[我的小红书主页](https://www.xiaohongshu.com/user/profile/63f11530000000001001e0c8?xhsshare=CopyLink&appuid=63f11530000000001001e0c8&apptime=1690528872)

💡 **支持我**：
- B站：[B站充电](https://space.bilibili.com/484366804)
- 爱发电：[为我充电](https://afdian.net/a/ZHOZHO)


## Credits

[Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3-api?utm_source=twitter&utm_medium=website&utm_campaign=blog)
