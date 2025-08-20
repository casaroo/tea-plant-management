// 茶树选择器组件
class TeaPlantSelector {
    constructor(options = {}) {
        this.areaSelectorId = options.areaSelectorId || 'areaSelector';
        this.plantSelectorId = options.plantSelectorId || 'plantSelector';
        this.refreshBtnId = options.refreshBtnId || 'refreshBtn';
        this.onPlantChange = options.onPlantChange || (() => {});
        this.onRefresh = options.onRefresh || (() => {});
        
        this.allPlants = [];
        this.currentPlantId = null;
        
        this.init();
    }
    
    async init() {
        try {
            await this.loadTeaPlants();
            this.bindEvents();
            console.log('茶树选择器初始化完成，茶树数量:', this.allPlants.length);
            // 初始化时根据当前区域选择一次，自动选中第一棵并触发回调
            this.filterPlants();
        } catch (error) {
            console.error('茶树选择器初始化失败:', error);
        }
    }
    
    async loadTeaPlants() {
        try {
            console.log('开始加载茶树列表...');
            const response = await fetch('/api/tea-plants');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            console.log('茶树数据响应:', data);
            
            if (data.plants && Array.isArray(data.plants)) {
                this.allPlants = data.plants;
                console.log(`成功加载 ${this.allPlants.length} 棵茶树`);
                this.populatePlantSelector();
            } else {
                console.error('茶树数据格式错误:', data);
                this.allPlants = [];
            }
        } catch (error) {
            console.error('加载茶树列表失败:', error);
            this.allPlants = [];
            // 显示错误信息给用户
            this.showError('加载茶树列表失败，请检查网络连接或刷新页面');
        }
    }
    
    showError(message) {
        const plantSelector = document.getElementById(this.plantSelectorId);
        if (plantSelector) {
            plantSelector.innerHTML = `<option value="">${message}</option>`;
        }
    }
    
    populatePlantSelector() {
        const plantSelector = document.getElementById(this.plantSelectorId);
        if (!plantSelector) {
            console.error('茶树选择器元素未找到:', this.plantSelectorId);
            return;
        }
        
        plantSelector.innerHTML = '<option value="">选择茶树...</option>';
        
        if (this.allPlants.length === 0) {
            plantSelector.innerHTML = '<option value="">暂无茶树数据</option>';
            return;
        }
        
        this.allPlants.forEach(plant => {
            const option = document.createElement('option');
            option.value = plant.id;
            const location = plant.location_area && plant.location_row 
                ? `${plant.location_area}${plant.location_row}` 
                : (plant.location || '未知位置');
            option.textContent = `${plant.plant_id || plant.id} - ${plant.variety} - ${location}`;
            plantSelector.appendChild(option);
        });
        
        console.log(`茶树选择器已填充 ${this.allPlants.length} 个选项`);
    }
    
    filterPlants() {
        const areaSelector = document.getElementById(this.areaSelectorId);
        const plantSelector = document.getElementById(this.plantSelectorId);
        
        if (!areaSelector || !plantSelector) {
            console.error('区域选择器或茶树选择器元素未找到');
            return;
        }
        
        const selectedArea = areaSelector.value;
        console.log('筛选区域:', selectedArea);
        
        // 清空当前选项
        plantSelector.innerHTML = '<option value="">选择茶树...</option>';
        
        // 根据区域筛选茶树
        const filteredPlants = selectedArea === 'all' 
            ? this.allPlants 
            : this.allPlants.filter(plant => {
                if (!plant.location_area) return false;
                const areaA = selectedArea;
                const areaB = selectedArea.endsWith('区') ? selectedArea.slice(0, -1) : `${selectedArea}区`;
                return plant.location_area === areaA || plant.location_area === areaB;
            });
        
        console.log(`筛选后茶树数量: ${filteredPlants.length}`);
        
        // 重新填充选项
        filteredPlants.forEach(plant => {
            const option = document.createElement('option');
            option.value = plant.id;
            const location = plant.location_area && plant.location_row 
                ? `${plant.location_area}${plant.location_row}` 
                : (plant.location || '未知位置');
            option.textContent = `${plant.plant_id || plant.id} - ${plant.variety} - ${location}`;
            plantSelector.appendChild(option);
        });
        
        // 重置当前选中的茶树
        if (filteredPlants.length > 0) {
            this.currentPlantId = filteredPlants[0].id;
            plantSelector.value = this.currentPlantId;
            console.log('自动选择第一棵茶树:', this.currentPlantId);
            this.onPlantChange(this.currentPlantId);
        } else {
            this.currentPlantId = null;
            this.onPlantChange(null);
        }
    }
    
    bindEvents() {
        const areaSelector = document.getElementById(this.areaSelectorId);
        const plantSelector = document.getElementById(this.plantSelectorId);
        const refreshBtn = document.getElementById(this.refreshBtnId);
        
        if (areaSelector) {
            areaSelector.addEventListener('change', () => {
                console.log('区域选择器变化:', areaSelector.value);
                this.filterPlants();
            });
        } else {
            console.error('区域选择器元素未找到:', this.areaSelectorId);
        }
        
        if (plantSelector) {
            plantSelector.addEventListener('change', (e) => {
                this.currentPlantId = e.target.value ? parseInt(e.target.value) : null;
                console.log('茶树选择器变化:', this.currentPlantId);
                this.onPlantChange(this.currentPlantId);
            });
        } else {
            console.error('茶树选择器元素未找到:', this.plantSelectorId);
        }
        
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                console.log('刷新按钮点击');
                this.onRefresh();
            });
        } else {
            console.error('刷新按钮元素未找到:', this.refreshBtnId);
        }
    }
    
    setCurrentPlant(plantId) {
        this.currentPlantId = plantId;
        const plantSelector = document.getElementById(this.plantSelectorId);
        if (plantSelector) {
            plantSelector.value = plantId || '';
        }
    }
    
    getCurrentPlant() {
        return this.currentPlantId;
    }
    
    getAllPlants() {
        return this.allPlants;
    }
    
    getPlantById(plantId) {
        return this.allPlants.find(plant => plant.id === plantId);
    }
    
    // 新增：手动刷新茶树列表
    async refreshTeaPlants() {
        await this.loadTeaPlants();
        this.filterPlants();
    }
}

// 导出供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TeaPlantSelector;
}
