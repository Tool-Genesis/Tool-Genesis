"""
动态令牌生成器模块
提供安全的动态令牌生成、验证和管理功能。
每个令牌都有唯一ID、过期时间和元数据。
"""

import secrets
import string
import time
import json
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class TokenInfo:
    """令牌信息数据类"""
    token_id: str
    token: str
    created_at: float
    expires_at: float
    metadata: Dict[str, Any]
    is_revoked: bool = False


class DynamicTokenGenerator:
    """
    动态令牌生成器
    
    提供安全的令牌生成、验证、撤销和管理功能。
    支持令牌过期、元数据存储和统计信息。
    """
    
    def __init__(self, default_length: int = 32, min_length: int = 16, max_length: int = 128):
        """
        初始化令牌生成器
        
        Args:
            default_length: 默认令牌长度
            min_length: 最小令牌长度
            max_length: 最大令牌长度
        """
        if default_length < min_length or default_length > max_length:
            raise ValueError(f"默认长度必须在 {min_length} 到 {max_length} 之间")
            
        self.default_length = default_length
        self.min_length = min_length
        self.max_length = max_length
        self.tokens: Dict[str, TokenInfo] = {}  # token -> TokenInfo
        self.token_ids: Dict[str, str] = {}     # token_id -> token
        
    def generate_token(
        self, 
        length: Optional[int] = None,
        expiry_minutes: int = 30,
        metadata: Optional[Dict[str, Any]] = None,
        server_name: Optional[str] = None
    ) -> str:
        """
        生成新的令牌
        
        Args:
            length: 令牌长度（如果为None则使用默认长度）
            expiry_minutes: 过期时间（分钟）
            metadata: 令牌元数据
            server_name: 服务器名称
            
        Returns:
            生成的令牌字符串
            
        Raises:
            ValueError: 当参数不合法时
        """
        # 验证长度参数
        if length is None:
            length = self.default_length
        elif length < self.min_length or length > self.max_length:
            raise ValueError(f"令牌长度必须在 {self.min_length} 到 {self.max_length} 之间")
        
        # 验证过期时间参数
        if expiry_minutes <= 0:
            raise ValueError("过期时间必须大于0分钟")
        if expiry_minutes > 10080:  # 7天
            raise ValueError("过期时间不能超过7天（10080分钟）")
        
        # 生成安全的随机令牌
        alphabet = string.ascii_letters + string.digits
        token = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        # 生成唯一的令牌ID
        token_id = hashlib.sha256(f"{token}_{time.time()}".encode()).hexdigest()[:16]
        
        # 计算过期时间
        current_time = time.time()
        expires_at = current_time + (expiry_minutes * 60)
        
        # 创建令牌信息
        token_info = TokenInfo(
            token_id=token_id,
            token=token,
            created_at=current_time,
            expires_at=expires_at,
            metadata=metadata or {},
            is_revoked=False
        )
        
        # 如果提供了服务器名称，添加到元数据中
        if server_name:
            token_info.metadata['server_name'] = server_name
        
        # 存储令牌信息
        self.tokens[token] = token_info
        self.token_ids[token_id] = token
        
        return token
    
    def verify_token(self, token: str) -> bool:
        """
        验证令牌是否有效
        
        Args:
            token: 要验证的令牌
            
        Returns:
            True如果令牌有效，False否则
        """
        token_info = self.tokens.get(token)
        if not token_info:
            return False
            
        # 检查是否被撤销
        if token_info.is_revoked:
            return False
            
        # 检查是否过期
        current_time = time.time()
        if current_time > token_info.expires_at:
            return False
            
        return True
    
    def validate_token(self, token: str) -> bool:
        """
        验证令牌是否有效（别名方法，保持向后兼容）
        
        Args:
            token: 要验证的令牌
            
        Returns:
            True如果令牌有效，False否则
        """
        return self.verify_token(token)
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        获取令牌信息
        
        Args:
            token: 令牌字符串
            
        Returns:
            令牌信息字典，如果令牌不存在则返回None
        """
        token_info = self.tokens.get(token)
        if not token_info:
            return None
            
        info_dict = asdict(token_info)
        # 转换时间戳为可读格式
        info_dict['created_at_readable'] = datetime.fromtimestamp(token_info.created_at).isoformat()
        info_dict['expires_at_readable'] = datetime.fromtimestamp(token_info.expires_at).isoformat()
        
        return info_dict
    
    def revoke_token(self, token: str) -> bool:
        """
        撤销令牌
        
        Args:
            token: 要撤销的令牌
            
        Returns:
            True如果撤销成功，False如果令牌不存在
        """
        token_info = self.tokens.get(token)
        if not token_info:
            return False
            
        token_info.is_revoked = True
        return True
    
    def revoke_token_by_id(self, token_id: str) -> bool:
        """
        通过ID撤销令牌
        
        Args:
            token_id: 令牌ID
            
        Returns:
            True如果撤销成功，False如果令牌不存在
        """
        token = self.token_ids.get(token_id)
        if not token:
            return False
            
        return self.revoke_token(token)
    
    def cleanup_expired_tokens(self) -> int:
        """
        清理过期的令牌
        
        Returns:
            清理的令牌数量
        """
        current_time = time.time()
        expired_tokens = []
        
        for token, token_info in self.tokens.items():
            if current_time > token_info.expires_at:
                expired_tokens.append(token)
        
        # 删除过期令牌
        for token in expired_tokens:
            token_info = self.tokens[token]
            del self.tokens[token]
            if token_info.token_id in self.token_ids:
                del self.token_ids[token_info.token_id]
        
        return len(expired_tokens)
    
    def set_token_expiry_policy(self, min_expiry_minutes: int = 1, max_expiry_minutes: int = 10080):
        """
        设置令牌过期策略
        
        Args:
            min_expiry_minutes: 最小过期时间（分钟）
            max_expiry_minutes: 最大过期时间（分钟）
        """
        if min_expiry_minutes <= 0:
            raise ValueError("最小过期时间必须大于0分钟")
        if max_expiry_minutes <= min_expiry_minutes:
            raise ValueError("最大过期时间必须大于最小过期时间")
        
        self.min_expiry_minutes = min_expiry_minutes
        self.max_expiry_minutes = max_expiry_minutes
    
    def get_active_tokens(self) -> List[Dict[str, Any]]:
        """
        获取所有活跃的令牌信息
        
        Returns:
            活跃令牌信息列表
        """
        current_time = time.time()
        active_tokens = []
        
        for token, token_info in self.tokens.items():
            if not token_info.is_revoked and current_time <= token_info.expires_at:
                info_dict = asdict(token_info)
                info_dict['created_at_readable'] = datetime.fromtimestamp(token_info.created_at).isoformat()
                info_dict['expires_at_readable'] = datetime.fromtimestamp(token_info.expires_at).isoformat()
                active_tokens.append(info_dict)
        
        return active_tokens
    
    def get_active_tokens_count(self) -> int:
        """获取活跃令牌数量"""
        current_time = time.time()
        count = 0
        for token_info in self.tokens.values():
            if not token_info.is_revoked and current_time <= token_info.expires_at:
                count += 1
        return count
    
    def get_token_count(self) -> Dict[str, int]:
        """
        获取令牌统计信息
        
        Returns:
            包含各种令牌计数的字典
        """
        current_time = time.time()
        total = len(self.tokens)
        active = 0
        expired = 0
        revoked = 0
        
        for token_info in self.tokens.values():
            if token_info.is_revoked:
                revoked += 1
            elif current_time > token_info.expires_at:
                expired += 1
            else:
                active += 1
        
        return {
            "total": total,
            "active": active,
            "expired": expired,
            "revoked": revoked
        }
    
    def clear_all_tokens(self) -> None:
        """清理所有令牌"""
        self.tokens.clear()
        self.token_ids.clear()


# 全局令牌生成器实例
_default_generator = DynamicTokenGenerator(default_length=32)


def generate_admin_token(
    expiry_minutes: int = 30, 
    metadata: Optional[Dict[str, Any]] = None,
    server_name: Optional[str] = None
) -> str:
    """
    生成管理员令牌的便捷函数
    
    Args:
        expiry_minutes: 过期时间（分钟）
        metadata: 令牌元数据
        server_name: 服务器名称
        
    Returns:
        生成的管理员令牌
    """
    return _default_generator.generate_token(
        expiry_minutes=expiry_minutes,
        metadata=metadata or {"purpose": "admin_access"},
        server_name=server_name
    )


def verify_admin_token(token: str) -> bool:
    """
    验证管理员令牌的便捷函数
    
    Args:
        token: 要验证的令牌
        
    Returns:
        True如果令牌有效，False否则
    """
    return _default_generator.verify_token(token)


def validate_admin_token(token: str) -> bool:
    """
    验证管理员令牌的便捷函数（别名方法，保持向后兼容）
    
    Args:
        token: 要验证的令牌
        
    Returns:
        True如果令牌有效，False否则
    """
    return _default_generator.verify_token(token)


def revoke_admin_token(token: str) -> bool:
    """
    撤销管理员令牌的便捷函数
    
    Args:
        token: 要撤销的令牌
        
    Returns:
        True如果撤销成功，False否则
    """
    return _default_generator.revoke_token(token)


def get_token_generator() -> DynamicTokenGenerator:
    """获取默认的令牌生成器实例"""
    return _default_generator


if __name__ == "__main__":
    # 测试代码
    print("=== 动态令牌生成器测试 ===")
    
    # 创建生成器
    generator = DynamicTokenGenerator()
    
    # 生成令牌
    token1 = generator.generate_token(expiry_minutes=1, metadata={"user": "admin"})
    print(f"生成令牌1: {token1}")
    
    token2 = generator.generate_token(expiry_minutes=2, metadata={"user": "user1"})
    print(f"生成令牌2: {token2}")
    
    # 验证令牌
    print(f"令牌1有效性: {generator.verify_token(token1)}")
    print(f"令牌2有效性: {generator.verify_token(token2)}")
    
    # 获取令牌信息
    info1 = generator.get_token_info(token1)
    print(f"令牌1信息: {info1}")
    
    # 撤销令牌
    print(f"撤销令牌1: {generator.revoke_token(token1)}")
    print(f"撤销后令牌1有效性: {generator.verify_token(token1)}")
    
    # 获取活跃令牌
    active_tokens = generator.get_active_tokens()
    print(f"活跃令牌数量: {len(active_tokens)}")
    
    # 获取统计信息
    stats = generator.get_token_count()
    print(f"令牌统计: {stats}")
    
    print("=== 测试完成 ===")