// AI量化交易学习平台 JavaScript

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('AI量化交易学习平台已加载');
    initializeApp();
});

// 初始化应用
function initializeApp() {
    // 检查系统状态
    checkSystemStatus();
    
    // 绑定事件监听器
    bindEventListeners();
    
    // 更新工作流程状态
    updateWorkflowStatus();
}

// 绑定事件监听器
function bindEventListeners() {
    // 这里可以添加更多的事件绑定
    console.log('事件监听器已绑定');
}

// 检查系统状态
function checkSystemStatus() {
    // 初始隐藏数据状态区域，等待数据加载后显示
    const dataStatus = document.getElementById('dataStatus');
    if (dataStatus) {
        dataStatus.style.display = 'none';
    }
    
    // 检查数据状态
    checkDataStatus();
    
    // 检查特征状态
    checkFeatureStatus();
    
    // 检查模型状态
    checkModelStatus();
    
    // 检查回测状态
    checkBacktestStatus();
}

// 下载数据
function downloadData() {
    const poolName = document.getElementById('stockPool').value;
    const dataSource = document.getElementById('dataSource').value;
    const startDate = document.getElementById('downloadStartDate').value;
    const endDate = document.getElementById('downloadEndDate').value;
    
    // 验证日期输入
    if (!startDate || !endDate) {
        alert('请选择开始和结束日期');
        return;
    }
    
    if (startDate >= endDate) {
        alert('开始日期必须早于结束日期');
        return;
    }
    
    console.log('开始下载数据:', { poolName, dataSource, startDate, endDate });
    
    document.getElementById('downloadProgress').style.display = 'block';
    document.getElementById('downloadStatus').textContent = `开始下载 (数据源: ${dataSource}, 时间: ${startDate} 到 ${endDate})...`;
    
    // 重置进度条
    const progressBar = document.getElementById('downloadProgressBar');
    progressBar.style.width = '0%';
    
    // 模拟进度 - 更真实的下载进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        // 模拟下载进度的不同阶段
        if (progress < 30) {
            progress += Math.random() * 8; // 开始阶段较慢
        } else if (progress < 70) {
            progress += Math.random() * 15; // 中间阶段较快
        } else if (progress < 90) {
            progress += Math.random() * 10; // 后期阶段中等
        } else {
            progress += Math.random() * 5; // 最后阶段较慢
        }
        
        if (progress > 95) progress = 95; // 最多到95%，等待实际完成
        
        progressBar.style.width = progress + '%';
        document.getElementById('downloadStatusText').textContent = `下载中... ${Math.round(progress)}%`;
    }, 300);
    
    fetch('/api/data/download', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            pool_name: poolName,
            data_source: dataSource,
            start_date: startDate.replace(/-/g, ''),
            end_date: endDate.replace(/-/g, '')
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 停止模拟进度
            clearInterval(progressInterval);
            
            // 确保进度条显示100%
            const progressBar = document.getElementById('downloadProgressBar');
            progressBar.style.width = '100%';
            
            // 更新状态显示
            document.getElementById('downloadStatus').textContent = '下载完成！';
            document.getElementById('downloadStatusText').textContent = '下载完成！';
            
            // 更新概览状态
            document.getElementById('downloadOverview').innerHTML = '<span class="badge bg-success">下载完成</span>';
            
            setTimeout(() => {
                checkDataStatus();
                updateWorkflowStep(1, 'completed');
                enableNextStep(2);
                
                // 保持进度条显示，但更新状态
                document.getElementById('downloadStatusText').textContent = '数据下载完成，可以进行下一步操作';
            }, 1000);
        } else {
            // 停止模拟进度
            clearInterval(progressInterval);
            alert('下载失败: ' + data.error);
            document.getElementById('downloadProgress').style.display = 'none';
        }
    })
    .catch(error => {
        // 停止模拟进度
        clearInterval(progressInterval);
        console.error('下载错误:', error);
        alert('网络错误: ' + error);
        document.getElementById('downloadProgress').style.display = 'none';
    });
}

// 构建特征
function buildFeatures() {
    console.log('开始构建特征');
    
    // 先检查是否有数据
    fetch('/api/data/status')
        .then(response => response.json())
        .then(data => {
            if (data.success && Object.keys(data.data.cleaned_data).length > 0) {
                // 有数据，开始构建特征
                startFeatureBuilding();
            } else {
                // 没有数据，提示用户
                alert('⚠️ 没有找到清洗后的数据！\n\n请先完成数据下载和清洗步骤。');
                return;
            }
        })
        .catch(error => {
            console.error('检查数据状态失败:', error);
            alert('检查数据状态失败，请重试');
        });
}

// 开始构建特征
function startFeatureBuilding() {
    document.getElementById('featureProgress').style.display = 'block';
    document.getElementById('featureStatus').textContent = '开始构建特征...';
    
    // 重置进度条
    const progressBar = document.getElementById('featureProgressBar');
    progressBar.style.width = '0%';
    
    // 模拟进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 25;
        if (progress > 100) progress = 100;
        progressBar.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            document.getElementById('featureStatus').textContent = '特征构建完成！';
        }
    }, 400);
    
    fetch('/api/features/build', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 确保进度条显示100%
            progressBar.style.width = '100%';
            document.getElementById('featureStatus').textContent = '特征构建完成！';
            
            setTimeout(() => {
                checkFeatureStatus();
                document.getElementById('featureProgress').style.display = 'none';
                updateWorkflowStep(2, 'completed');
                enableNextStep(3);
            }, 2000); // 显示2秒完成状态
        } else {
            clearInterval(progressInterval); // 停止模拟进度
            alert('特征构建失败: ' + data.error);
            document.getElementById('featureProgress').style.display = 'none';
        }
    })
    .catch(error => {
        clearInterval(progressInterval); // 停止模拟进度
        console.error('特征构建错误:', error);
        alert('网络错误: ' + error);
        document.getElementById('featureProgress').style.display = 'none';
    });
}

// 训练模型
function trainModels() {
    console.log('开始训练模型');
    
    // 先检查是否有特征数据
    fetch('/api/features/status')
        .then(response => response.json())
        .then(data => {
            const features = (data && data.data) ? data.data : {};
            if (data.success && Object.keys(features).length > 0) {
                // 有特征数据，开始训练模型
                startModelTraining();
            } else {
                // 没有特征数据，提示用户
                alert('⚠️ 没有找到特征数据！\n\n请先完成特征构建步骤。');
                return;
            }
        })
        .catch(error => {
            console.error('检查特征状态失败:', error);
            alert('检查特征状态失败，请重试');
        });
}

// 开始训练模型
function startModelTraining() {
    document.getElementById('trainingProgress').style.display = 'block';
    document.getElementById('trainingStatus').textContent = '开始训练模型...';
    
    // 重置进度条
    const progressBar = document.getElementById('trainingProgressBar');
    progressBar.style.width = '0%';
    
    // 模拟进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 30;
        if (progress > 100) progress = 100;
        progressBar.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            document.getElementById('trainingStatus').textContent = '模型训练完成！';
        }
    }, 300);
    
    fetch('/api/models/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            cv_folds: parseInt(document.getElementById('cvFolds').value) || 5,
            random_seed: parseInt(document.getElementById('randomSeed').value) || 42,
            start_date: document.getElementById('trainingStartDate').value.replace(/-/g, ''),
            end_date: document.getElementById('trainingEndDate').value.replace(/-/g, '')
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 确保进度条显示100%
            progressBar.style.width = '100%';
            document.getElementById('trainingStatus').textContent = '模型训练完成！';
            
            setTimeout(() => {
                checkModelStatus();
                document.getElementById('trainingProgress').style.display = 'none';
                updateWorkflowStep(3, 'completed');
                enableNextStep(4);
            }, 2000); // 显示2秒完成状态
        } else {
            clearInterval(progressInterval); // 停止模拟进度
            alert('模型训练失败: ' + data.error);
            document.getElementById('trainingProgress').style.display = 'none';
        }
    })
    .catch(error => {
        clearInterval(progressInterval); // 停止模拟进度
        console.error('模型训练错误:', error);
        alert('网络错误: ' + error);
        document.getElementById('trainingProgress').style.display = 'none';
    });
}

// 运行回测
function runBacktest() {
    console.log('开始运行回测');
    
    // 先检查是否有训练好的模型
    fetch('/api/models/status')
        .then(response => response.json())
        .then(data => {
            const models = (data && data.data) ? data.data : {};
            if (data.success && Object.keys(models).length > 0) {
                // 有模型，开始回测
                startBacktest();
            } else {
                // 没有模型，提示用户
                alert('⚠️ 没有找到训练好的模型！\n\n请先完成模型训练步骤。');
                return;
            }
        })
        .catch(error => {
            console.error('检查模型状态失败:', error);
            alert('检查模型状态失败，请重试');
        });
}

// 开始回测
function startBacktest() {
    document.getElementById('backtestProgress').style.display = 'block';
    document.getElementById('backtestStatus').textContent = '开始运行回测...';
    
    // 重置进度条
    const progressBar = document.getElementById('backtestProgressBar');
    progressBar.style.width = '0%';
    
    // 模拟进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 35;
        if (progress > 100) progress = 100;
        progressBar.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            document.getElementById('backtestStatus').textContent = '回测完成！';
        }
    }, 250);
    
    // 保存进度条引用，以便后续停止
    window.backtestProgressInterval = progressInterval;
    
    fetch('/api/backtest/run', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            cash: parseFloat(document.getElementById('backtestCash').value) || 100000,
            commission: parseFloat(document.getElementById('backtestCommission').value) || 0.0008,
            ml_threshold: parseFloat(document.getElementById('backtestMlThreshold').value) || 0.55,
            start_date: document.getElementById('backtestStartDate').value.replace(/-/g, ''),
            end_date: document.getElementById('backtestEndDate').value.replace(/-/g, '')
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 确保进度条显示100%
            progressBar.style.width = '100%';
            document.getElementById('backtestStatus').textContent = '回测完成！';
            
            // 立即停止模拟进度
            if (window.backtestProgressInterval) {
                clearInterval(window.backtestProgressInterval);
                window.backtestProgressInterval = null;
            }
            
            setTimeout(() => {
                checkBacktestStatus();
                document.getElementById('backtestProgress').style.display = 'none';
                updateWorkflowStep(4, 'completed');
                enableNextStep(5);
                showBacktestResults(data.results);
            }, 2000); // 显示2秒完成状态
        } else {
            if (window.backtestProgressInterval) {
                clearInterval(window.backtestProgressInterval);
                window.backtestProgressInterval = null;
            }
            alert('回测失败: ' + data.error);
            document.getElementById('backtestProgress').style.display = 'none';
        }
    })
    .catch(error => {
        if (window.backtestProgressInterval) {
            clearInterval(window.backtestProgressInterval);
            window.backtestProgressInterval = null;
        }
        console.error('回测错误:', error);
        alert('网络错误: ' + error);
        document.getElementById('backtestProgress').style.display = 'none';
    });
}

// 运行历史验证
function runHistoricalValidation() {
    console.log('开始运行历史验证');
    
    // 先检查是否有训练好的模型
    fetch('/api/models/status')
        .then(response => response.json())
        .then(data => {
            const models = (data && data.data) ? data.data : {};
            if (data.success && Object.keys(models).length > 0) {
                // 有模型，开始历史验证
                startHistoricalValidation();
            } else {
                // 没有模型，提示用户
                alert('⚠️ 没有找到训练好的模型！\n\n请先完成模型训练步骤。');
                return;
            }
        })
        .catch(error => {
            console.error('检查模型状态失败:', error);
            alert('检查模型状态失败，请重试');
        });
}

// 开始历史验证
function startHistoricalValidation() {
    document.getElementById('historicalProgress').style.display = 'block';
    document.getElementById('historicalStatus').textContent = '开始运行历史验证...';
    
    // 重置进度条
    const progressBar = document.getElementById('historicalProgressBar');
    progressBar.style.width = '0%';
    
    // 模拟进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 40;
        if (progress > 100) progress = 100;
        progressBar.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            document.getElementById('historicalStatus').textContent = '历史验证完成！';
        }
    }, 200);
    
    // 获取用户设置的验证时间范围
    const startDate = document.getElementById('historicalStartDate').value;
    const endDate = document.getElementById('historicalEndDate').value;
    
    // 验证日期输入
    if (!startDate || !endDate) {
        alert('请设置验证的开始和结束日期');
        return;
    }
    
    if (startDate >= endDate) {
        alert('开始日期必须早于结束日期');
        return;
    }
    
    fetch('/api/historical/run', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            start_date: startDate,
            end_date: endDate
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 确保进度条显示100%
            progressBar.style.width = '100%';
            document.getElementById('historicalStatus').textContent = '历史验证完成！';
            document.getElementById('historicalStatusText').textContent = '历史验证完成！';
            
            setTimeout(() => {
                document.getElementById('historicalProgress').style.display = 'none';
                updateWorkflowStep(5, 'completed');
                showHistoricalResults(data.results);
            }, 2000); // 显示2秒完成状态
        } else {
            clearInterval(progressInterval); // 停止模拟进度
            alert('历史验证失败: ' + data.error);
            document.getElementById('historicalProgress').style.display = 'none';
        }
    })
    .catch(error => {
        clearInterval(progressInterval); // 停止模拟进度
        console.error('历史验证错误:', error);
        alert('网络错误: ' + error);
        document.getElementById('historicalProgress').style.display = 'none';
    });
}

// 运行未来预测
function runFuturePrediction() {
    console.log('开始运行未来预测');
    
    // 先检查是否有训练好的模型
    fetch('/api/models/status')
        .then(response => response.json())
        .then(data => {
            const models = (data && data.data) ? data.data : {};
            if (data.success && Object.keys(models).length > 0) {
                // 有模型，开始未来预测
                startFuturePrediction();
            } else {
                // 没有模型，提示用户
                alert('⚠️ 没有找到训练好的模型！\n\n请先完成模型训练步骤。');
                return;
            }
        })
        .catch(error => {
            console.error('检查模型状态失败:', error);
            alert('检查模型状态失败，请重试');
        });
}

// 开始未来预测
function startFuturePrediction() {
    const stockCode = document.getElementById('futureStockCode').value.trim();
    const predictionDays = parseInt(document.getElementById('predictionDays').value) || 5;
    const confidenceThreshold = parseFloat(document.getElementById('confidenceThreshold').value) || 0.6;
    
    if (!stockCode) {
        alert('请输入股票代码！');
        return;
    }
    
    document.getElementById('futureProgress').style.display = 'block';
    document.getElementById('futureStatus').textContent = '开始运行未来预测...';
    
    // 重置进度条
    const progressBar = document.getElementById('futureProgressBar');
    progressBar.style.width = '0%';
    
    // 模拟进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 40;
        if (progress > 100) progress = 100;
        progressBar.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            document.getElementById('futureStatus').textContent = '未来预测完成！';
        }
    }, 200);
    
    fetch('/api/future/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            stock_code: stockCode,
            prediction_days: predictionDays,
            confidence_threshold: confidenceThreshold
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 确保进度条显示100%
            progressBar.style.width = '100%';
            document.getElementById('futureStatus').textContent = '未来预测完成！';
            document.getElementById('futureStatusText').textContent = '未来预测完成！';
            
            setTimeout(() => {
                document.getElementById('futureProgress').style.display = 'none';
                updateWorkflowStep(6, 'completed');
                showFutureResults(data.results);
            }, 2000); // 显示2秒完成状态
        } else {
            clearInterval(progressInterval); // 停止模拟进度
            alert('未来预测失败: ' + data.error);
            document.getElementById('futureProgress').style.display = 'none';
        }
    })
    .catch(error => {
        clearInterval(progressInterval); // 停止模拟进度
        console.error('未来预测错误:', error);
        alert('网络错误: ' + error);
        document.getElementById('futureProgress').style.display = 'none';
    });
}

// 清空所有数据
function clearAllData() {
    console.log('开始清空所有数据');
    
    // 二次确认
    const confirmMessage = `⚠️ 危险操作确认

此操作将删除：
• 所有股票数据
• 所有特征数据  
• 所有训练好的模型
• 所有回测结果
• 所有预测结果

确定要清空所有数据吗？此操作不可撤销！`;

    if (!confirm(confirmMessage)) {
        console.log('用户取消清空操作');
        return;
    }
    
    // 显示进度
    document.getElementById('clearDataProgress').style.display = 'block';
    document.getElementById('clearDataBtn').disabled = true;
    document.getElementById('clearDataBtn').innerHTML = '<i class="bi bi-hourglass-split"></i> 清空中...';
    
    // 重置进度条
    const progressBar = document.getElementById('clearDataProgressBar');
    progressBar.style.width = '0%';
    
    // 模拟进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 30;
        if (progress > 90) progress = 90; // 最多到90%，等待实际完成
        progressBar.style.width = progress + '%';
    }, 100);
    
    // 调用清空API
    fetch('/api/data/clear', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            confirm: true
        })
    })
    .then(response => response.json())
    .then(data => {
        // 停止进度模拟
        clearInterval(progressInterval);
        
        if (data.success) {
            // 显示完成状态
            progressBar.style.width = '100%';
            document.getElementById('clearDataStatusText').textContent = '数据清空完成！';
            
            // 更新状态显示
            document.getElementById('clearDataOverview').innerHTML = `
                <span class="badge bg-success">清空完成</span>
                <small class="text-muted d-block mt-2">
                    清空了 ${data.cleared_directories} 个目录，总大小: ${data.total_size_mb} MB
                </small>
            `;
            
            // 重置所有模块状态
            resetAllModules();
            
            // 重置工作流程
            resetWorkflow();
            
            setTimeout(() => {
                document.getElementById('clearDataProgress').style.display = 'none';
                document.getElementById('clearDataBtn').disabled = false;
                document.getElementById('clearDataBtn').innerHTML = '<i class="bi bi-trash"></i> 清空所有数据';
                
                // 显示成功消息
                alert(`✅ 数据清空完成！\n\n清空了 ${data.cleared_directories} 个目录\n总大小: ${data.total_size_mb} MB\n\n现在可以重新开始整个流程了！`);
            }, 2000);
            
        } else {
            clearInterval(progressInterval);
            alert('❌ 数据清空失败: ' + data.error);
            document.getElementById('clearDataProgress').style.display = 'none';
            document.getElementById('clearDataBtn').disabled = false;
            document.getElementById('clearDataBtn').innerHTML = '<i class="bi bi-trash"></i> 清空所有数据';
        }
    })
    .catch(error => {
        clearInterval(progressInterval);
        console.error('清空数据错误:', error);
        alert('网络错误: ' + error);
        document.getElementById('clearDataProgress').style.display = 'none';
        document.getElementById('clearDataBtn').disabled = false;
        document.getElementById('clearDataBtn').innerHTML = '<i class="bi bi-trash"></i> 清空所有数据';
    });
}

// 重置所有模块状态
function resetAllModules() {
    // 重置数据模块
    document.getElementById('downloadStatus').innerHTML = '<p class="text-muted">等待开始下载</p>';
    document.getElementById('downloadOverview').innerHTML = '<span class="badge bg-secondary">等待下载</span>';
    
    // 重置特征模块
    document.getElementById('featureStatus').innerHTML = '<p class="text-muted">等待数据下载完成</p>';
    document.getElementById('featureOverview').innerHTML = '<span class="badge bg-secondary">等待构建</span>';
    
    // 重置训练模块
    document.getElementById('trainingStatus').innerHTML = '<p class="text-muted">等待特征构建完成</p>';
    document.getElementById('trainingOverview').innerHTML = '<span class="badge bg-secondary">等待训练</span>';
    
    // 重置回测模块
    document.getElementById('backtestStatus').innerHTML = '<p class="text-muted">等待模型训练完成</p>';
    document.getElementById('backtestOverview').innerHTML = '<span class="badge bg-secondary">等待回测</span>';
    
    // 重置历史验证模块
    document.getElementById('historicalStatus').innerHTML = '<p class="text-muted">等待模型训练完成</p>';
    document.getElementById('historicalOverview').innerHTML = '<span class="badge bg-secondary">等待验证</span>';
    
    // 重置未来预测模块
    document.getElementById('futureStatus').innerHTML = '<p class="text-muted">等待模型训练完成</p>';
    document.getElementById('futureOverview').innerHTML = '<span class="badge bg-secondary">等待预测</span>';
    
    // 隐藏所有结果区域
    const resultAreas = ['dataStatus', 'featureStatusDisplay', 'trainingResults', 'backtestResults', 'historicalResults', 'futureResults'];
    resultAreas.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = 'none';
        }
    });
}

// 重置工作流程
function resetWorkflow() {
    // 重置所有步骤为初始状态
    for (let i = 1; i <= 6; i++) {
        const stepElement = document.getElementById(`step${i}`);
        if (stepElement) {
            stepElement.classList.remove('active', 'completed');
        }
    }
    
    // 设置第一步为激活状态
    const step1 = document.getElementById('step1');
    if (step1) {
        step1.classList.add('active');
    }
}

// 更新工作流程步骤
function updateWorkflowStep(stepNumber, status) {
    const stepElement = document.getElementById(`step${stepNumber}`);
    if (stepElement) {
        // 移除所有状态类
        stepElement.classList.remove('active', 'completed');
        // 添加新状态类
        stepElement.classList.add(status);
    }
}

// 启用下一步 (现在主要用于更新工作流程状态)
function enableNextStep(stepNumber) {
    // 按钮现在总是可用的，这里主要用于更新工作流程状态
    console.log(`步骤 ${stepNumber} 已完成，可以继续下一步`);
}

// 检查数据状态
function checkDataStatus() {
    fetch('/api/data/status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateDataStatus(data.data);
            }
        })
        .catch(error => console.error('检查数据状态失败:', error));
}

// 检查特征状态
function checkFeatureStatus() {
    fetch('/api/features/status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateFeatureStatus(data.data);
            }
        })
        .catch(error => console.error('检查特征状态失败:', error));
}

// 检查模型状态
function checkModelStatus() {
    fetch('/api/models/status')
        .then(response => response.json())
        .then(data => {
            const models = (data && data.data) ? data.data : {};
            if (data.success) {
                updateModelStatus(models);
            }
        })
        .catch(error => console.error('检查模型状态失败:', error));
}

// 检查回测状态
function checkBacktestStatus() {
    fetch('/api/backtest/status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateBacktestStatus(data.data);
            }
        })
        .catch(error => console.error('检查回测状态失败:', error));
}

// 更新状态显示函数
function updateDataStatus(data) {
    // 计算数据数量
    const rawCount = Object.keys(data.raw_data || {}).length;
    const cleanedCount = Object.keys(data.cleaned_data || {}).length;
    
    // 更新概览显示
    const overview = document.getElementById('dataOverview');
    if (overview) {
        if (rawCount > 0 || cleanedCount > 0) {
            overview.innerHTML = `
                <span class="badge bg-primary">原始: ${rawCount} 只股票</span>
                <span class="badge bg-success">清洗: ${cleanedCount} 只股票</span>
            `;
        } else {
            overview.innerHTML = `
                <span class="badge bg-secondary">等待数据</span>
            `;
        }
    }
    
    // 同时更新数据状态显示区域
    const dataStatus = document.getElementById('dataStatus');
    if (dataStatus && (rawCount > 0 || cleanedCount > 0)) {
        dataStatus.style.display = 'block';
        
        let statusHTML = '<div class="row">';
        
        if (rawCount > 0) {
            statusHTML += `
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title text-primary">
                                <i class="bi bi-download"></i> 原始数据
                            </h6>
                            <p class="card-text">已下载 ${rawCount} 只股票的数据</p>
                            <small class="text-muted">包含价格、成交量等原始信息</small>
                        </div>
                    </div>
                </div>
            `;
        }
        
        if (cleanedCount > 0) {
            statusHTML += `
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h6 class="card-title text-success">
                                <i class="bi bi-check-circle"></i> 清洗数据
                            </h6>
                            <p class="card-text">已清洗 ${cleanedCount} 只股票的数据</p>
                            <small class="text-muted">去除异常值，数据质量提升</small>
                        </div>
                    </div>
                </div>
            `;
        }
        
        statusHTML += '</div>';
        dataStatus.querySelector('.row').innerHTML = statusHTML;
    }
}

function updateFeatureStatus(data) {
    const overview = document.getElementById('featureOverview');
    if (overview) {
        const count = Object.keys(data || {}).length;
        
        overview.innerHTML = `
            <span class="badge bg-success">${count} 只股票</span>
        `;
    }
}

function updateModelStatus(data) {
    const overview = document.getElementById('modelOverview');
    if (overview) {
        const count = Object.keys(data || {}).length;
        
        overview.innerHTML = `
            <span class="badge bg-warning">${count} 个模型</span>
        `;
    }
    
    // 显示详细的模型状态
    const modelStatus = document.getElementById('modelStatus');
    if (modelStatus && Object.keys(data || {}).length > 0) {
        modelStatus.style.display = 'block';
        
        let statusHTML = '<div class="row">';
        
        for (const [stock, modelInfo] of Object.entries(data)) {
            statusHTML += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">
                                <i class="bi bi-robot"></i> ${stock}
                            </h6>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-6">
                                    <small class="text-muted">训练日期</small><br>
                                    <strong>${modelInfo.training_date}</strong>
                                </div>
                                <div class="col-6">
                                    <small class="text-muted">特征数量</small><br>
                                    <strong>${modelInfo.feature_count}</strong>
                                </div>
                            </div>
                            <hr>
                            <div class="row">
                                <div class="col-6">
                                    <small class="text-muted">交叉验证准确率</small><br>
                                    <strong class="text-primary">${modelInfo.cv_accuracy}</strong>
                                </div>
                                <div class="col-6">
                                    <small class="text-muted">测试集F1分数</small><br>
                                    <strong class="text-success">${modelInfo.test_f1}</strong>
                                </div>
                            </div>
                            <hr>
                            <div class="row">
                                <div class="col-6">
                                    <small class="text-muted">训练集准确率</small><br>
                                    <strong class="text-info">${modelInfo.train_accuracy}</strong>
                                </div>
                                <div class="col-6">
                                    <small class="text-muted">测试集准确率</small><br>
                                    <strong class="text-warning">${modelInfo.test_accuracy}</strong>
                                </div>
                            </div>
                            <hr>
                            <div class="row">
                                <div class="col-6">
                                    <small class="text-muted">训练样本数</small><br>
                                    <strong>${modelInfo.training_samples}</strong>
                                </div>
                                <div class="col-6">
                                    <small class="text-muted">测试样本数</small><br>
                                    <strong>${modelInfo.test_samples}</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        statusHTML += '</div>';
        modelStatus.querySelector('.row').innerHTML = statusHTML;
    }
}

function updateBacktestStatus(data) {
    const overview = document.getElementById('backtestOverview');
    if (overview) {
        const count = Object.keys(data || {}).length;
        
        if (count > 0) {
            overview.innerHTML = `
                <span class="badge bg-success">${count} 次回测</span>
            `;
        } else {
            overview.innerHTML = `
                <span class="badge bg-secondary">等待回测</span>
            `;
        }
    }
}

// 显示回测结果
function showBacktestResults(results) {
    const resultsDiv = document.getElementById('backtestResults');
    if (!resultsDiv) return;
    
    // 显示结果区域
    resultsDiv.style.display = 'block';
    
    let html = '<h5 class="mb-3">回测结果概览</h5>';
    html += '<div class="row">';
    
    let successCount = 0;
    let totalCount = 0;
    
    for (const [stock, result] of Object.entries(results)) {
        totalCount++;
        
        if (result.ml_strategy && result.baseline_strategy) {
            successCount++;
            const ml = result.ml_strategy;
            const base = result.baseline_strategy;
            
            // 安全地处理数值，避免NaN或undefined
            const mlReturn = isNaN(ml.total_return) ? 0 : ml.total_return;
            const mlSharpe = isNaN(ml.sharpe) ? 0 : ml.sharpe;
            const baseReturn = isNaN(base.total_return) ? 0 : base.total_return;
            const baseSharpe = isNaN(base.sharpe) ? 0 : base.sharpe;
            
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">${stock} 回测结果</h6>
                        </div>
                        <div class="card-body">
                            <p><strong>ML策略:</strong> 总收益 ${(mlReturn * 100).toFixed(2)}%, 夏普比率 ${mlSharpe.toFixed(3)}</p>
                            <p><strong>基线策略:</strong> 总收益 ${(baseReturn * 100).toFixed(2)}%, 夏普比率 ${baseSharpe.toFixed(3)}</p>
                        </div>
                    </div>
                </div>
            `;
        } else if (result.status === 'error') {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card border-danger">
                        <div class="card-header text-danger">
                            <h6 class="mb-0">${stock} 回测失败</h6>
                        </div>
                        <div class="card-body">
                            <p class="text-danger">错误: ${result.error || '未知错误'}</p>
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    html += '</div>';
    
    // 添加统计信息
    html += `
        <div class="mt-3">
            <div class="alert alert-info">
                <strong>回测统计:</strong> 成功 ${successCount}/${totalCount} 只股票
            </div>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
}

// 显示预测结果
function showPredictionResults(results) {
    const resultsDiv = document.getElementById('predictionResults');
    if (!resultsDiv) return;
    
    resultsDiv.style.display = 'block';
    
    let html = '<h5 class="mb-3">预测结果概览</h5>';
    html += '<div class="row">';
    
    let successCount = 0;
    let totalCount = 0;
    
    for (const [stock, result] of Object.entries(results)) {
        totalCount++;
        
        if (result.success && result.predictions_summary) {
            successCount++;
            const summary = result.predictions_summary;
            const signals = result.signals || [];
            
            // 统计交易信号
            const buySignals = signals.filter(s => s.action === 'BUY').length;
            const sellSignals = signals.filter(s => s.action === 'SELL').length;
            const holdSignals = signals.filter(s => s.action === 'HOLD').length;
            
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">${stock} 预测结果</h6>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-6">
                                    <p><strong>数据范围:</strong> ${summary.date_range}</p>
                                    <p><strong>总样本:</strong> ${summary.total_records}</p>
                                    <p><strong>平均概率:</strong> ${(summary.avg_probability * 100).toFixed(1)}%</p>
                                </div>
                                <div class="col-6">
                                    <p><strong>上涨预测:</strong> ${summary.up_predictions} (${(summary.up_ratio * 100).toFixed(1)}%)</p>
                                    <p><strong>下跌预测:</strong> ${summary.down_predictions} (${((1-summary.up_ratio) * 100).toFixed(1)}%)</p>
                                    <p><strong>概率范围:</strong> ${(summary.min_probability * 100).toFixed(1)}% - ${(summary.max_probability * 100).toFixed(1)}%</p>
                                </div>
                            </div>
                            <hr>
                            <div class="row">
                                <div class="col-4 text-center">
                                    <span class="badge bg-success">买入: ${buySignals}</span>
                                </div>
                                <div class="col-4 text-center">
                                    <span class="badge bg-danger">卖出: ${sellSignals}</span>
                                </div>
                                <div class="col-4 text-center">
                                    <span class="badge bg-secondary">观望: ${holdSignals}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else if (result.success === false) {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card border-danger">
                        <div class="card-header text-danger">
                            <h6 class="mb-0">${stock} 预测失败</h6>
                        </div>
                        <div class="card-body">
                            <p class="text-danger">错误: ${result.error || '未知错误'}</p>
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    html += '</div>';
    
    // 添加统计信息
    html += `
        <div class="mt-3">
            <div class="alert alert-info">
                <strong>预测统计:</strong> 成功 ${successCount}/${totalCount} 只股票
            </div>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
}

// 更新工作流程状态
function updateWorkflowStatus() {
    // 这里可以根据实际状态更新工作流程
    console.log('工作流程状态已更新');
}

// 显示历史验证结果
function showHistoricalResults(results) {
    const resultsDiv = document.getElementById('historicalResults');
    if (!resultsDiv) return;
    
    resultsDiv.style.display = 'block';
    
    let html = '<h5 class="mb-3">历史验证结果概览</h5>';
    html += '<div class="row">';
    
    let successCount = 0;
    let totalCount = 0;
    
    for (const [stock, result] of Object.entries(results)) {
        totalCount++;
        
        if (result.success) {
            successCount++;
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">${stock} 历史验证结果</h6>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-6">
                                    <p><strong>验证样本数:</strong> ${result.samples}</p>
                                    <p><strong>交易信号数:</strong> ${result.signals}</p>
                                    <p><strong>验证时间:</strong> ${result.validation_period?.start || 'N/A'} 到 ${result.validation_period?.end || 'N/A'}</p>
                                </div>
                                <div class="col-6">
                                    <p><strong>状态:</strong> <span class="badge bg-success">验证成功</span></p>
                                    <p><strong>备注:</strong> 使用2025年数据验证模型效果</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card border-danger">
                        <div class="card-header text-danger">
                            <h6 class="mb-0">${stock} 验证失败</h6>
                        </div>
                        <div class="card-body">
                            <p class="text-danger">错误: ${result.error || '未知错误'}</p>
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    html += '</div>';
    
    // 添加统计信息
    html += `
        <div class="mt-3">
            <div class="alert alert-info">
                <strong>历史验证统计:</strong> 成功 ${successCount}/${totalCount} 只股票
            </div>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
}

// 显示未来预测结果
function showFutureResults(results) {
    const resultsDiv = document.getElementById('futureResults');
    if (!resultsDiv) return;
    
    resultsDiv.style.display = 'block';
    
    let html = '<h5 class="mb-3">未来预测结果</h5>';
    
    if (results && results.predictions) {
        const predictions = results.predictions;
        html += `
            <div class="card mb-3">
                <div class="card-body">
                    <h6 class="card-title">${results.stock_code} - 未来${results.total_predictions}天预测</h6>
                    <p><strong>预测时间:</strong> ${results.prediction_date}</p>
                    
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>日期</th>
                                    <th>预测</th>
                                    <th>概率</th>
                                    <th>信号</th>
                                    <th>置信度</th>
                                </tr>
                            </thead>
                            <tbody>
        `;
        
        for (const pred of predictions) {
            const actionClass = pred.action === 'BUY' ? 'text-success' : 
                               pred.action === 'SELL' ? 'text-danger' : 'text-warning';
            
            html += `
                <tr>
                    <td>${pred.date}</td>
                    <td>${pred.prediction === 1 ? '📈 上涨' : '📉 下跌'}</td>
                    <td>${(pred.probability * 100).toFixed(1)}%</td>
                    <td><span class="${actionClass}">${pred.emoji} ${pred.action}</span></td>
                    <td>${(pred.confidence * 100).toFixed(1)}%</td>
                </tr>
            `;
        }
        
        html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    } else {
        html += '<div class="alert alert-info">暂无未来预测结果</div>';
    }
    
    resultsDiv.innerHTML = html;
}
