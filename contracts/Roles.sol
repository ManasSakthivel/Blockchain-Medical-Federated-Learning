// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title Roles
 * @dev Library for managing addresses assigned to a Role.
 */
library Roles {
    struct Role {
        mapping (address => bool) bearer;
    }

    /**
     * @dev Give an account access to this role.
     */
    function add(Role storage role, address account) internal {
        require(account != address(0), "Roles: account is the zero address");
        require(!has(role, account), "Roles: account already has role");
        role.bearer[account] = true;
    }

    /**
     * @dev Remove an account's access to this role.
     */
    function remove(Role storage role, address account) internal {
        require(has(role, account), "Roles: account does not have role");
        role.bearer[account] = false;
    }

    /**
     * @dev Check if an account has this role. The zero address never has a
     *      role (returns false rather than reverting, so view helpers such as
     *      `isAuthorized(address(0))` are safe to call).
     * @return bool
     */
    function has(Role storage role, address account) internal view returns (bool) {
        if (account == address(0)) return false;
        return role.bearer[account];
    }
} 