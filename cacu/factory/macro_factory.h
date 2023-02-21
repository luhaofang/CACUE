#ifndef MACRO_FACTORY
#define MACRO_FACTORY


#include <string>
#include <map>


using namespace std;

namespace cacu{

class macro_factory
{
public:

	struct register_macro{

		register_macro(const string& key, const op_name& op_name_){
			auto ito = macro_factory::instance()->m_value2key.find(op_name_);
			if (ito != macro_factory::instance()->m_value2key.end())
				LOG_FATAL("Op type %d is already registered for %s!", op_name_, ito->second.c_str());
			auto itk = macro_factory::instance()->m_key2value.find(key);
			if (itk != macro_factory::instance()->m_key2value.end())
				LOG_FATAL("Op name %s is already registered as %d!", key.c_str(), itk->second);
			macro_factory::instance()->m_value2key[op_name_] = key;
			macro_factory::instance()->m_key2value[key] = op_name_;
			macro_factory::instance()->m_index2op[int(op_name_)] = op_name_;
		}
	};

	struct register_name{

		register_name(const string& cname, const string& tname){
			auto ito = macro_factory::instance()->m_tname2cname.find(tname);
			if (ito != macro_factory::instance()->m_tname2cname.end())
				LOG_FATAL("Op type %s is already registered for %s!", tname.c_str(), ito->second.c_str());
			macro_factory::instance()->m_tname2cname[tname] = cname;
		}
	};

    static macro_factory* instance()
    {
        static macro_factory f;
    	return &f;
    }

    auto key2value(const string& key) -> op_name
	{
		op_name re = CACU_NULL;
		auto ite = m_key2value.find(key);
		if (ite != m_key2value.end()) {
			re = (ite->second);
		}
		if(re == CACU_NULL)
			LOG_FATAL("Can't find the op named %s!", key.c_str());
		return re;
	}

    auto value2key(const op_name& op_name_) -> string
	{
		string re = "";
		auto ite = m_value2key.find(op_name_);
		if (ite != m_value2key.end()) {
			re = (ite->second);
		}
		if(re == "")
			LOG_FATAL("Can't find the op name by %d!", op_name_);
		return re;
	}

    auto index2op(const int index_) -> op_name
	{
    	op_name re = CACU_NULL;
		auto ite = m_index2op.find(index_);
		if (ite != m_index2op.end()) {
			re = (ite->second);
		}
		if(re == CACU_NULL)
			LOG_FATAL("Can't find the op index of %d!", index_);
		return re;
	}

    auto tname2cname(const string& op_name_) -> string
	{
		string re = "";
		auto ite = m_tname2cname.find(op_name_);
		if (ite != m_tname2cname.end()) {
			re = (ite->second);
		}
		if(re == "")
			LOG_FATAL("Can't find the op name by %s!", op_name_.c_str());
		return re;
	}

    static auto get_op_type(const string& key) -> op_name
    {
        return macro_factory::instance()->key2value(key);
    }

    static auto get_op_name(const op_name& op_name_) -> string
    {
	    return macro_factory::instance()->value2key(op_name_);
    }

    static auto get_optype_by_index(const int index_) -> op_name
	{
    	return macro_factory::instance()->index2op(index_);
	}

    static auto get_cname(const string& tname) -> string
	{
		return macro_factory::instance()->tname2cname(tname);
	}

private:
    macro_factory() {}
    macro_factory(const macro_factory&) = delete;
    macro_factory(macro_factory&&) = delete;
    macro_factory& operator =(const macro_factory&) = delete;

    std::map<op_name, string> m_value2key;
    std::map<string, op_name> m_key2value;
    std::map<int, op_name> m_index2op;
    std::map<string, string> m_tname2cname;


};

}

#endif // MACRO_FACTORY
