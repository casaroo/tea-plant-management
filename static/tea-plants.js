// 茶树生长阶段切换功能
function switchGrowthStage(stage) {
    // 隐藏所有阶段内容
    const allStages = document.querySelectorAll('[id$="-stage"]');
    allStages.forEach(stageDiv => {
        stageDiv.style.display = 'none';
    });
    
    // 显示选中的阶段
    const targetStage = document.getElementById(stage + '-stage');
    if (targetStage) {
        targetStage.style.display = 'block';
    }
    
    // 更新按钮状态
    const buttons = document.querySelectorAll('#stage-buttons button');
    buttons.forEach(btn => {
        btn.classList.remove('bg-green-600', 'text-white');
        btn.classList.add('bg-gray-100', 'text-gray-700', 'hover:bg-gray-200');
    });
    
    // 激活当前按钮
    const activeButton = document.querySelector(`[data-stage="${stage}"]`);
    if (activeButton) {
        activeButton.classList.remove('bg-gray-100', 'text-gray-700', 'hover:bg-gray-200');
        activeButton.classList.add('bg-green-600', 'text-white');
    }
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    // 为所有阶段按钮添加点击事件监听器
    const buttons = document.querySelectorAll('#stage-buttons button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            const stage = this.getAttribute('data-stage');
            switchGrowthStage(stage);
        });
    });
    
    // 默认显示幼苗期
    switchGrowthStage('seedling');
});