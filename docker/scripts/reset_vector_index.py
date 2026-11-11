#!/usr/bin/env python3
"""
重置向量库索引脚本，用于解决维度不匹配问题
注意：执行此脚本会删除现有向量索引并重建，所有已索引的数据会丢失！
"""
import os
import sys

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

# 加载环境变量
env_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(env_path)

from src.applications.rag.script.setup_es_collections import (
    create_es_client,
    _resolve_collection_names,
    _resolve_embedding_dim,
    init_es_collections_from_env
)

def main():
    print("⚠️  警告：此操作将删除现有向量索引并重建，所有已索引的数据会丢失！")
    confirm = input("请输入 'YES' 确认继续操作: ")
    if confirm != 'YES':
        print("操作已取消")
        return
    
    # 获取配置
    es_host = os.getenv('ES_HOST', 'localhost:9200')
    base_collection = os.getenv('VECTOR_COLLECTION', 'yu_rag')
    
    # 创建ES客户端
    client = create_es_client(es_host)
    
    try:
        # 检查ES连接
        if not client.ping():
            print(f"❌ 无法连接到 Elasticsearch: {es_host}")
            return
        
        print(f"✅ 成功连接到 Elasticsearch: {es_host}")
        
        # 获取所有相关集合名称
        children, parents, grandchildren = _resolve_collection_names(base=base_collection)
        all_collections = [children, parents, grandchildren]
        print(f"\n找到相关索引: {', '.join(all_collections)}")
        
        # 检查并删除现有索引
        deleted = []
        for idx_name in all_collections:
            if client.indices.exists(index=idx_name):
                print(f"正在删除索引: {idx_name}")
                client.indices.delete(index=idx_name)
                deleted.append(idx_name)
            else:
                print(f"索引不存在，跳过: {idx_name}")
        
        if deleted:
            print(f"\n✅ 已删除索引: {', '.join(deleted)}")
        else:
            print("\nℹ️  没有需要删除的索引")
        
        # 获取当前配置的维度
        dim = _resolve_embedding_dim()
        print(f"\n当前配置的向量维度: {dim}")
        
        # 重新创建索引
        print("\n正在创建新的向量索引...")
        init_es_collections_from_env(
            es_host=es_host, 
            dim=dim,
            base=base_collection,
            parents=os.getenv('VECTOR_PARENTS_COLLECTION'),
            grandchildren=os.getenv('VECTOR_GRANDCHILDREN_COLLECTION')
        )
        
        # 验证索引创建成功
        print("\n验证索引创建结果:")
        for idx_name in all_collections:
            if client.indices.exists(index=idx_name):
                mapping = client.indices.get_mapping(index=idx_name)
                vector_dim = mapping[idx_name]['mappings']['properties']['vector']['dims']
                print(f"✅ 索引 {idx_name} 创建成功，维度: {vector_dim}")
            else:
                print(f"❌ 索引 {idx_name} 创建失败")
        
        print("\n🎉 向量索引重置完成！")
        print("现在可以重启服务，向量维度不匹配的问题应该已经解决。")
        print("注意：你需要重新上传和索引所有文档。")
        
    except Exception as e:
        print(f"\n❌ 操作失败: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    main()
