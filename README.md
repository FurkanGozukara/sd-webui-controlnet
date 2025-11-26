# 1 Click Installer for Automatic1111 SD Web UI, SDXL, ControlNet, All ControlNet Models, Full xFormers and Latest Libraries Support for RTX 5000 Series GPUs and All Other GPUs With Torch 2.8 and CUDA 12.9

## https://www.patreon.com/posts/86307255

## This repo is only made for SECourses Patreon Members use Above Link to Install

[Patreon exclusive posts index](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Patreon-Posts-Index.md) *to find our scripts easily,* [Patreon scripts updates history](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Generative-AI-Updates-And-News.md) *to see which updates arrived to which scripts and amazing* [Patreon special generative scripts list](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Amazing-Generative-AI-Scripts.md) *that you can use in any of your task.*

*Join discord to get help, chat, discuss and also tell me your discord username to get your special rank :* [SECourses Discord](https://discord.com/servers/software-engineering-courses-secourses-772774097734074388)

*Please also Star, Watch and Fork our* [Stable Diffusion & Generative AI](https://github.com/FurkanGozukara/Stable-Diffusion) *GitHub repository and join our* [Reddit subreddit](https://www.reddit.com/r/SECourses/) *and follow me on* [LinkedIn](https://www.linkedin.com/in/furkangozukara/) *(my real profile)*

**=======**

Latest installer zip file : https://www.patreon.com/posts/86307255

*   I have forked the official and latest version of Automatic1111 : [https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
    *   ADetailer : [https://github.com/Bing-su/adetailer](https://github.com/Bing-su/adetailer)
    *   ControlNet : [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
    *   Installation process is now way faster than before
*   Automatic1111 is very old but there is still people demanding and using it so here it is
    *   But I would recommend still up-to-date developed NEO
        *   Classic : [https://www.patreon.com/posts/138680643](https://www.patreon.com/posts/138680643)
        *   Forge Web UI NEO : [https://www.patreon.com/posts/138694680](https://www.patreon.com/posts/138694680)
*   Then modified all of them with minimal amount to make them work with latest GPUs and xFormers
    *   Now our installer and apps supports starting from RTX 1000 series to RTX 5000 series including cloud GPUs like H100, B200, RTX PRO 6000 etc
    *   Making them work with newest libraries were so hard but now we are using Torch 2.8, CUDA 12.9, xFormers 0.33
*   The installer will install ADetailer and ControlNet by default but you have to download ControlNet models manually or run Windows_Optional_Download_ControlNet_Models.bat file after install
*   Default Sampler set to UniPc and default steps count set to 40
*   Tested on RTX 5090 and works perfect and amazingly fast with xFormers
*   The app will start with xFormers by default both on Linux (MassedCompute and RunPod) and Windows
    *   On MassedCompute and RunPod it will also auto start with Gradio live share to access from your computer
*   Put Stable Diffusion models into stable-diffusion-webui\models\Stable-diffusion
*   ControlNet models into stable-diffusion-webui\extensions\sd-webui-controlnet\models
*   The app won't download any model like before when you first time start and you can start the app without loading any model - this was not possible before i fixed both

### **Windows Requirements**

*   Python 3.10, FFmpeg, CUDA 12.9, cuDNN 9.12 , C++ tools and Git
*   If it doesn't work make sure to below tutorial and install everything exactly as shown in this below tutorial
*   [https://youtu.be/DrhUHnYfwC0](https://youtu.be/DrhUHnYfwC0)

### **How To Install and Use:**

### **Windows:**

*   Just run Windows_Install_or_Update.bat
*   Then run Windows_Start_Automatic1111_Web_UI.bat

### **Massed Compute (Recommend Cloud) :**

*   Please register via this link : [https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute](https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute)
    *   Use our coupon SECourses
    *   Our coupon works on all GPUs now
        *   RTX 6000 PRO has amazing price and speed but you can use like RTX A6000 ADA or L40S as well
        *   Full details here : [https://www.patreon.com/posts/26671823](https://www.patreon.com/posts/26671823)
    *   Then select our image SECourses from Creator dropdown
    *   Then follow Massed_Compute_Instructions_READ.txt
    *   Same as my any other Massed Compute installer script
    *   Example tutorial for learn how to install and use Massed Compute
        *   (Starts at 12:58) : [https://youtu.be/KW-MHmoNcqo?si=G1WbG-Qw4ujWvOtG&t=778](https://youtu.be/KW-MHmoNcqo?si=G1WbG-Qw4ujWvOtG&t=778)

### **RunPod (Cloud):**

*   Please register via this link : [https://get.runpod.io/955rkuppqv4h](https://get.runpod.io/955rkuppqv4h)
    *   Then follow Runpod_Instructions_READ.txt
    *   Same as my any other RunPod installer script
    *   Use the template written in Runpod_Instructions_READ.txt file
    *   Example tutorial for learn how to install and use RunPod
        *   (starts at 22:03) : [https://youtu.be/KW-MHmoNcqo?si=QN8X8Sjn13ZYu-EU&t=1323](https://youtu.be/KW-MHmoNcqo?si=QN8X8Sjn13ZYu-EU&t=1323)


<img width="3840" height="5043" alt="screencapture-127-0-0-1-7860-2025-11-27-00_41_48" src="https://github.com/user-attachments/assets/c0d940d9-a45b-4f18-a703-12b51cd85c85" />

<img width="3842" height="4524" alt="screencapture-127-0-0-1-7860-2025-11-27-02_14_26" src="https://github.com/user-attachments/assets/4cdf18da-4c05-4c9e-9434-9de24724996a" />

<img width="1776" height="1048" alt="modelss" src="https://github.com/user-attachments/assets/a5c1de27-9d67-435d-b579-865c711ca1c4" />

<img width="2000" height="2064" alt="supported" src="https://github.com/user-attachments/assets/684655a5-86b3-4bef-bff9-e02dd16eaf2d" />


